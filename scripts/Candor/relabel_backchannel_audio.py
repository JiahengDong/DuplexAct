#!/usr/bin/env python3
"""
Audio-based backchannel-type relabeler for Floor-Holding + Backchannel candidates.

Takes the already GPT-validated candidates from:
    results/floor_backchannel_backbiter/valid.json

For each candidate it sends the AUDIO to GPT-4o-audio-preview so the model can
judge the prosodic and acoustic cues of B's backchannel — information the text-only
pass cannot see.  Only `backchannel_type` is re-classified; `floor_holding_label`
and validity are left unchanged.

Audio sent per candidate
------------------------
  full_stereo.wav   — T0-start → T1-end, both channels, no silencing.
                      Gives GPT full conversational context + the backchannel in situ.
  backchannel.wav   — B's channel across the whole of T1 (no silencing).
                      Highlights the backchannel in isolation.

Output
------
  results/floor_backchannel_backbiter/
    valid_audio_relabeled.json          ← full updated candidate list
    audio_relabel_log.json              ← per-candidate before/after + reason
    <floor_label>/<bc_type_audio>/      ← re-organised by NEW audio label
        candidates.json
        <audio files> (symlinked or copied)

Usage
-----
  python relabel_backchannel_audio.py
  python relabel_backchannel_audio.py --dry_run        # print first 3 prompts, no API
  python relabel_backchannel_audio.py --limit 20       # process only first N
  python relabel_backchannel_audio.py --skip_existing  # skip already labelled entries
"""

import argparse
import base64
import json
import os
import shutil
import time
from pathlib import Path

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
GPT_MODEL      = "gpt-audio-1.5"

CANDOR_ROOT  = Path("/data/gpfs/projects/punim2960/jiahengd/dataset/Candor")
RESULTS_DIR  = Path("/data/gpfs/projects/punim2960/jiahengd/results/floor_backchannel_backbiter")
VALID_JSON   = RESULTS_DIR / "valid.json"
OUT_JSON     = RESULTS_DIR / "valid_audio_relabeled.json"
LOG_JSON     = RESULTS_DIR / "audio_relabel_log.json"

FLOOR_HOLDING_LABELS = ["continue_current", "continue_shorten", "continue_simplify", "none"]
BACKCHANNEL_TYPES    = ["acknowledgement", "understanding", "agreement",
                        "uncertainty", "disagreement", "none"]

BACKCHANNEL_TYPE_GUIDANCE = {
    "acknowledgement": (
        "A minimal 'I'm still listening / received' signal that conveys attention "
        "without committing to a stance — e.g. 'mhm', 'uh huh', 'right', 'okay'. It "
        "keeps the channel open but does not claim comprehension, agreement, or doubt."
    ),
    "understanding": (
        "Signals that the listener has grasped/followed the content — e.g. 'oh I see', "
        "'got it', 'ah okay', 'that makes sense'. Stronger than bare acknowledgement: "
        "it claims comprehension of what A is saying."
    ),
    "agreement": (
        "Signals alignment or endorsement of A's point — e.g. 'yeah exactly', 'totally', "
        "'for sure', 'absolutely', 'so true'. The listener is not just following but "
        "actively agreeing with the stance/claim. Only valid when A made a claim that "
        "could be agreed with."
    ),
    "uncertainty": (
        "Signals hesitation, doubt, or being unsure/confused — e.g. 'hmm', 'I'm not "
        "sure', 'really?', 'wait', a doubtful 'uh'. The listener is following but not "
        "convinced or not fully clear."
    ),
    "disagreement": (
        "Signals pushback, objection, or non-alignment — e.g. 'no', 'I don't think so', "
        "'mm not really', 'eh'. The listener indicates they do not accept A's point."
    ),
    "none": (
        "The backchannel does not clearly fit any of the above types — e.g. it is "
        "ambiguous, multi-functional, or its social meaning is unclear from context. "
        "Use this as a last resort only when no other type is a reasonable fit."
    ),
}


def _bc_block() -> str:
    lines = []
    for label, desc in BACKCHANNEL_TYPE_GUIDANCE.items():
        lines.append(f'  "{label}": {desc}')
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are an expert in conversational analysis and prosodics. You will be given:
  1. Two audio clips from a real two-person conversation.
  2. A text transcript of the relevant segment.

Your task is to classify the TYPE of a single backchannel produced by Speaker B \
while Speaker A is holding the floor.

A backchannel is a brief signal (often non-lexical) from a listener \
that does NOT claim the floor. You must judge based on BOTH the acoustic/prosodic \
cues in the audio AND the semantic context in the transcript.

Pay special attention to:
  - Pitch contour (rising = uncertainty/question, falling = agreement/acknowledgement)
  - Vowel quality and duration (drawn-out vowels often signal understanding or \
surprise; clipped ones signal bare attention)
  - Intensity relative to surrounding speech
  - The content A was delivering just before the backchannel (claim vs. neutral info)

Respond ONLY with JSON — no markdown, no prose outside the JSON object."""

PROMPT_TEMPLATE = """\
=== Conversation context ===
Speaker B (trigger turn, Turn 0):
  "{t0_text}"

Speaker A (floor-holding turn, Turn 1):
  BEFORE backchannel: "{before_text}"
  [BACKCHANNEL by B: "{bc_text}"  at {bc_start:.2f}–{bc_end:.2f}s within Turn 1]
  AFTER backchannel:  "{after_text}"

=== Audio clips ===
Audio 1 (full_stereo): Both speakers, from start of Turn 0 to end of Turn 1.
Audio 2 (backchannel_channel): Speaker B's microphone only, across all of Turn 1 \
(the backchannel is audible in context here).

=== Task ===
Classify the backchannel type. Choose EXACTLY ONE from:

{bc_block}

Respond ONLY with JSON:
{{
  "backchannel_type": "<one of the labels above>",
  "reason": "one sentence referencing both acoustic and semantic evidence"
}}"""


# ── Audio helpers ─────────────────────────────────────────────────────────────

def find_audio_files(cand: dict) -> tuple[Path | None, Path | None]:
    """Return (full_stereo_path, backchannel_path) for a candidate.

    Search order:
      1. Results directory (nested by floor_label / bc_type_text_label)
      2. Local per-conversation audio_dir field
      3. Glob fallback in CANDOR_ROOT
    """
    base       = f"{cand['conv_id']}_t{cand['t1_turn_id']}"
    floor_lbl  = cand.get("floor_holding_label", "unknown")
    bc_lbl     = cand.get("backchannel_type", "unknown")

    # 1. Results dir (nested structure)
    results_subdir = RESULTS_DIR / floor_lbl / bc_lbl
    stereo_r  = results_subdir / f"{base}_full_stereo.wav"
    bc_r      = results_subdir / f"{base}_backchannel.wav"
    if stereo_r.exists() and bc_r.exists():
        return stereo_r, bc_r

    # 2. audio_dir field
    if cand.get("audio_dir"):
        adir = Path(cand["audio_dir"])
        stereo_a = adir / f"{base}_full_stereo.wav"
        bc_a     = adir / f"{base}_backchannel.wav"
        if stereo_a.exists() and bc_a.exists():
            return stereo_a, bc_a

    # 3. Glob fallback
    conv_dirs = list(CANDOR_ROOT.glob(f"*/{cand['conv_id']}"))
    if conv_dirs:
        adir = conv_dirs[0] / "floor_backchannel_backbiter" / "audio" / bc_lbl
        stereo_g = adir / f"{base}_full_stereo.wav"
        bc_g     = adir / f"{base}_backchannel.wav"
        if stereo_g.exists() and bc_g.exists():
            return stereo_g, bc_g

    return None, None


def encode_audio(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── GPT-4o audio call ─────────────────────────────────────────────────────────

def classify_with_audio(cand: dict, client: OpenAI,
                        dry_run: bool = False) -> dict | None:
    """
    Returns a dict with keys: backchannel_type, reason.
    Returns None on failure.
    """
    stereo_path, bc_path = find_audio_files(cand)
    if stereo_path is None or bc_path is None:
        return None

    user_text = PROMPT_TEMPLATE.format(
        t0_text    = cand.get("t0_text", ""),
        before_text= cand.get("before_text", ""),
        bc_text    = cand.get("bc_text", ""),
        bc_start   = cand.get("bc_start", 0),
        bc_end     = cand.get("bc_end", 0),
        after_text = cand.get("after_text", ""),
        bc_block   = _bc_block(),
    )

    if dry_run:
        print(f"\n{'─'*60}")
        print(f"CANDIDATE: {cand['conv_id']} t{cand['t1_turn_id']}")
        print(f"  text_label : {cand.get('backchannel_type')}")
        print(f"  stereo     : {stereo_path}")
        print(f"  bc_channel : {bc_path}")
        print(f"  PROMPT:\n{user_text[:600]}...")
        return None

    stereo_b64 = encode_audio(stereo_path)
    bc_b64     = encode_audio(bc_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": stereo_b64, "format": "wav"}},
                {"type": "input_audio", "input_audio": {"data": bc_b64,    "format": "wav"}},
                {"type": "text",        "text": user_text},
            ],
        },
    ]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            bc_type = parsed.get("backchannel_type", "").strip().lower()
            if bc_type not in BACKCHANNEL_TYPES:
                bc_type = "none"
            return {"backchannel_type": bc_type, "reason": parsed.get("reason", "")}
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    GPT error after 3 attempts: {e}")
                return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-label backchannel_type using GPT-4o audio input.")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Preview prompts for first --limit candidates, no API calls.")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Process at most N candidates.")
    parser.add_argument("--skip_existing",action="store_true",
                        help="Skip candidates already in valid_audio_relabeled.json.")
    parser.add_argument("--valid_json",   type=Path, default=VALID_JSON,
                        help="Path to source valid.json")
    parser.add_argument("--results_dir",  type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    # Load candidates
    data: list[dict] = json.load(open(args.valid_json))
    print(f"Loaded {len(data)} candidates from {args.valid_json}")

    # Load existing relabeled results to support --skip_existing
    existing: dict[str, dict] = {}
    if args.skip_existing and OUT_JSON.exists():
        for c in json.load(open(OUT_JSON)):
            key = f"{c['conv_id']}_t{c['t1_turn_id']}"
            existing[key] = c
        print(f"  Skipping {len(existing)} already relabeled candidates.")

    if args.limit:
        data = data[:args.limit]

    client = OpenAI(api_key=OPENAI_API_KEY) if not args.dry_run else None

    updated: list[dict] = []
    log:     list[dict] = []
    n_changed = 0
    n_failed  = 0
    n_skipped = 0

    for i, cand in enumerate(data):
        key = f"{cand['conv_id']}_t{cand['t1_turn_id']}"

        if args.skip_existing and key in existing:
            updated.append(existing[key])
            n_skipped += 1
            continue

        old_type = cand.get("backchannel_type", "none")
        print(f"[{i+1:3d}/{len(data)}] {cand['conv_id'][:8]}… t{cand['t1_turn_id']}"
              f"  text_label={old_type}", end="  ", flush=True)

        result = classify_with_audio(cand, client, dry_run=args.dry_run)

        if args.dry_run:
            updated.append(cand)
            continue

        if result is None:
            # audio not found or API failure — keep original label
            print("SKIP (no audio or API error)")
            updated.append(cand)
            n_failed += 1
            log.append({"key": key, "status": "failed",
                        "text_label": old_type, "audio_label": None})
            continue

        new_type = result["backchannel_type"]
        changed  = new_type != old_type
        if changed:
            n_changed += 1

        print(f"→ {new_type}" + (" (CHANGED)" if changed else ""))

        updated_cand = {**cand,
                        "backchannel_type_text":  old_type,
                        "backchannel_type":        new_type,
                        "backchannel_type_audio_reason": result["reason"]}
        updated.append(updated_cand)
        log.append({"key":        key,
                    "status":     "ok",
                    "text_label": old_type,
                    "audio_label":new_type,
                    "changed":    changed,
                    "reason":     result["reason"]})

    if args.dry_run:
        print("\nDry run complete — no files written.")
        return

    # ── Write updated valid.json ─────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump(updated, f, indent=2)

    with open(LOG_JSON, "w") as f:
        json.dump(log, f, indent=2)

    # ── Re-organise results dir by new audio label ───────────────────────────
    new_results = args.results_dir / "audio_relabeled"
    if new_results.exists():
        shutil.rmtree(new_results)
    new_results.mkdir(parents=True)

    by_floor: dict[str, list] = {}
    for c in updated:
        fl  = c.get("floor_holding_label", "unknown")
        bct = c.get("backchannel_type", "unknown")
        by_floor.setdefault(fl, {}).setdefault(bct, []).append(c)   # type: ignore[index]

    for fl, bc_map in by_floor.items():
        fl_dir = new_results / fl
        fl_dir.mkdir(parents=True, exist_ok=True)
        fl_entries = [c for entries in bc_map.values() for c in entries]
        with open(fl_dir / "candidates.json", "w") as f:
            json.dump(fl_entries, f, indent=2)

        for bct, entries in bc_map.items():
            subdir = fl_dir / bct
            subdir.mkdir(parents=True, exist_ok=True)
            with open(subdir / "candidates.json", "w") as f:
                json.dump(entries, f, indent=2)
            for cand in entries:
                base    = f"{cand['conv_id']}_t{cand['t1_turn_id']}"
                # Locate audio from original text-label location
                old_bct = cand.get("backchannel_type_text", bct)
                old_fl  = fl
                src_dir = args.results_dir / old_fl / old_bct
                for suffix in ("_t0.wav", "_t1_full.wav",
                               "_backchannel.wav", "_full_stereo.wav"):
                    src = src_dir / f"{base}{suffix}"
                    if src.exists():
                        shutil.copy2(src, subdir / f"{base}{suffix}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Relabeling complete")
    print(f"  Total      : {len(data)}")
    print(f"  Skipped    : {n_skipped}")
    print(f"  Failed     : {n_failed}")
    print(f"  Changed    : {n_changed}")
    print(f"\nNew backchannel_type distribution (audio-based):")
    from collections import defaultdict
    bc_counts: dict[str, int] = defaultdict(int)
    for c in updated:
        bc_counts[c.get("backchannel_type","unknown")] += 1
    for t in BACKCHANNEL_TYPES:
        old_n = sum(1 for c in data if c.get("backchannel_type")==t)
        new_n = bc_counts.get(t, 0)
        print(f"  {t:<25}  text={old_n:3d}  audio={new_n:3d}")
    print(f"\nOutputs:")
    print(f"  {OUT_JSON}")
    print(f"  {LOG_JSON}")
    print(f"  {new_results}/")


if __name__ == "__main__":
    main()
