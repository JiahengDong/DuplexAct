#!/usr/bin/env python3
"""
Batch processor for CANDOR interruption extraction.

Every timing and GPT validity criterion can be toggled off individually.
Pass the same --skip_* / --min_* flags here as you would to the extractor;
they are forwarded to each per-conversation subprocess.

Usage:
  python batch_interruptions.py --workers 8 --no_gpt \\
      --skip_abab --skip_t1_pauses --skip_t1_sent_final --min_t3_overlap 0.1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CANDOR_ROOT         = Path(os.environ.get("CANDOR_ROOT", "/data/gpfs/projects/punim2960/jiahengd/dataset/Candor"))
SCRIPT              = Path(__file__).resolve().parent / "extract_interruptions.py"
PYTHON              = sys.executable
DEFAULT_RESULTS_DIR = Path("/data/gpfs/projects/punim2960/jiahengd/results/interruptions_relaxed")

CONV_SUBDIR_DEFAULT = "interruption_candidates_relaxed"
QUOTA_EXIT_CODE    = 2
TARGET_PER_TYPE    = 500

TIMING_LABELS: dict[str, str] = {
    "windows_checked":    "Windows checked",
    "not_abab":           "Not A-B-A-B alternation",
    "t2_overlaps_t1":     "T2 starts before T1 ends",
    "t3_overlap_small":   "T3 overlap too small",
    "t4_overlaps_t3":     "T4 overlaps T3",
    "t1_no_sent_final":   "T1 no sentence-final punct",
    "t1_few_words":       "T1 too few words",
    "t1_has_pauses":      "T1 has internal pauses",
    "t1_has_backchannel": "T1 has backchannel from B",
    "t2_is_question":     "T2 is a question",
    "t2_few_words":       "T2 too few words",
    "t2_pre_words_short": "T2 pre-overlap words too short",
    "t2_has_backchannel": "T2 has backchannel from A",
    "t3_has_backchannel": "T3 has backchannel from B",
    "t3_has_pauses":      "T3 has internal pauses",
    "t3_too_long":        "T3 too long",
    "t3_empty":           "T3 empty",
    "t4_few_words":       "T4 too few words",
    "duplicate":          "Duplicate",
    "passed":             "PASSED timing",
}


def find_conversations(parts: list[int] | None = None) -> list[Path]:
    convs = []
    for part_dir in sorted(CANDOR_ROOT.glob("part_*")):
        if parts:
            if int(part_dir.name.split("_")[1]) not in parts:
                continue
        for conv_dir in sorted(part_dir.iterdir()):
            if conv_dir.is_dir() and (conv_dir / "transcription" / "transcript_backbiter.csv").exists():
                convs.append(conv_dir)
    return convs


def is_done(conv_dir: Path, no_gpt: bool = False,
            conv_subdir: str = CONV_SUBDIR_DEFAULT) -> bool:
    if no_gpt:
        marker = f"{conv_subdir}/filter1_timing/all.json"
    else:
        marker = f"{conv_subdir}/filter2_gpt/all_with_gpt_decisions.json"
    return (conv_dir / marker).exists()


def process_one(conv_dir: Path, results_dir: Path | None,
                no_gpt: bool, extra_args: list[str]) -> dict:
    start = time.time()
    cmd = [PYTHON, str(SCRIPT), "--conv_dir", str(conv_dir)]
    if results_dir:
        cmd += ["--results_dir", str(results_dir)]
    if no_gpt:
        cmd.append("--no_gpt")
    cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - start, 1)

    quota_hit = result.returncode == QUOTA_EXIT_CODE
    success   = result.returncode == 0

    timing_found  = 0
    valid_found   = 0
    timing_stats: dict = {}
    gpt_stats:    dict = {}
    for line in result.stdout.splitlines():
        if "Candidates found:" in line:
            try: timing_found = int(line.split(":")[1].strip())
            except: pass
        elif "Valid after GPT" in line:
            try: valid_found = int(line.split(":")[1].split("/")[0].strip())
            except: pass
        elif line.startswith("TIMING_STATS:"):
            try: timing_stats = json.loads(line[len("TIMING_STATS:"):])
            except: pass
        elif line.startswith("GPT_STATS:"):
            try: gpt_stats = json.loads(line[len("GPT_STATS:"):])
            except: pass

    return {
        "conv_dir":     str(conv_dir),
        "success":      success,
        "quota_hit":    quota_hit,
        "elapsed":      elapsed,
        "timing_found": timing_found,
        "valid_found":  valid_found,
        "timing_stats": timing_stats,
        "gpt_stats":    gpt_stats,
        "stderr":       result.stderr[-500:] if not success else "",
    }


def aggregate_results(results_dir: Path, all_convs: list[Path],
                      no_gpt: bool = False,
                      target_per_type: int = TARGET_PER_TYPE,
                      conv_subdir: str = CONV_SUBDIR_DEFAULT) -> None:
    import shutil
    from collections import defaultdict

    if no_gpt:
        all_candidates: list[dict] = []
        for conv_dir in all_convs:
            cpath = conv_dir / conv_subdir / "filter1_timing" / "all.json"
            if cpath.exists():
                try:
                    all_candidates.extend(json.load(open(cpath)))
                except Exception:
                    pass
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "timing_candidates.json", "w") as f:
            json.dump(all_candidates, f, indent=2)
        print(f"\n── Timing-only results aggregated → {results_dir} ──")
        print(f"  Total timing candidates: {len(all_candidates)}")
        return

    all_valid: list[dict] = []
    for conv_dir in all_convs:
        vpath = conv_dir / conv_subdir / "filter2_gpt" / "valid.json"
        if vpath.exists():
            try:
                all_valid.extend(json.load(open(vpath)))
            except Exception:
                pass

    counts_by_type: dict[str, int] = defaultdict(int)
    capped: list[dict] = []
    for c in all_valid:
        itype = c.get("interruption_type") or "unknown"
        if counts_by_type[itype] < target_per_type:
            capped.append(c)
            counts_by_type[itype] += 1
    if len(capped) < len(all_valid):
        print(f"  Capped: {len(all_valid)} → {len(capped)} (target={target_per_type}/type)")
    all_valid = capped

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not all_valid:
        print("\nNo valid candidates to aggregate.")
        return

    with open(results_dir / "valid.json", "w") as f:
        json.dump(all_valid, f, indent=2)

    for style in ("hard",):
        subset = [c for c in all_valid if c.get("interruption_style") == style]
        if subset:
            with open(results_dir / f"{style}.json", "w") as f:
                json.dump(subset, f, indent=2)

    by_style_type: dict[str, list] = defaultdict(list)
    for c in all_valid:
        style = c.get("interruption_style", "unknown")
        itype = c.get("interruption_type",  "unknown")
        by_style_type[f"{style}/{itype}"].append(c)

    for key, entries in by_style_type.items():
        style, itype = key.split("/", 1)
        subdir = results_dir / style / itype
        subdir.mkdir(parents=True, exist_ok=True)
        with open(subdir / "candidates.json", "w") as f:
            json.dump(entries, f, indent=2)
        for cand in entries:
            conv_id    = cand["conv_id"]
            base       = f"{conv_id}_t{cand['t1_turn_id']}"
            local_audio = cand.get("audio_dir")
            if local_audio:
                local_dir = Path(local_audio)
            else:
                conv_dirs = list(CANDOR_ROOT.glob(f"*/{conv_id}"))
                if not conv_dirs:
                    continue
                local_dir = conv_dirs[0] / conv_subdir / "audio" / style / itype
            for suffix in ("_t1.wav", "_t2.wav", "_t3.wav", "_t4.wav", "_full_stereo.wav"):
                src = local_dir / f"{base}{suffix}"
                if src.exists():
                    import shutil
                    shutil.copy2(src, subdir / f"{base}{suffix}")

    print(f"\n── Global results aggregated → {results_dir} ──")
    print(f"  Total valid (capped at {target_per_type}/type): {len(all_valid)}")
    for key, entries in sorted(by_style_type.items()):
        print(f"    {key}: {len(entries)}")


def main():
    global CANDOR_ROOT
    parser = argparse.ArgumentParser(
        description="Batch interruption extraction.")

    # ── Core options ───────────────────────────────────────────────────────────
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--parts",       type=int, nargs="+")
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--no_gpt",      action="store_true",
                        help="Run timing filter only; skip GPT classification and audio export.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--candor_root", type=Path, default=CANDOR_ROOT)
    parser.add_argument("--target",      type=int, default=TARGET_PER_TYPE,
                        help=f"Max valid candidates per recovery type (default: {TARGET_PER_TYPE})")

    # ── Pre-filter threshold (forwarded to extractor) ─────────────────────────
    parser.add_argument("--min_words_filter", type=int, default=None,
                        help="Pre-filter: drop turns with fewer than N words (0=off)")

    # ── Timing criteria (forwarded to extractor) ──────────────────────────────
    parser.add_argument("--skip_abab",          action="store_true")
    parser.add_argument("--skip_t2_overlap_t1", action="store_true")
    parser.add_argument("--max_t2_t1_overlap",  type=float, default=1.5,
                        help="Max T2-before-T1 overlap allowed (default 1.5s; 0.05=strict)")
    parser.add_argument("--skip_t3_overlap",    action="store_true")
    parser.add_argument("--min_t3_overlap",     type=float, default=0.3,
                        help="Minimum T3/T2 overlap in seconds (default 0.3)")
    parser.add_argument("--skip_t4_overlap_t3", action="store_true")
    parser.add_argument("--max_t4_t3_overlap",  type=float, default=1.5,
                        help="Max T4-before-T3 overlap allowed (default 1.5s; 0.05=strict)")
    parser.add_argument("--skip_t1_sent_final", action="store_true")
    parser.add_argument("--skip_t1_words",      action="store_true")
    parser.add_argument("--skip_t1_pauses",       action="store_true")
    parser.add_argument("--skip_t1_backchannel",  action="store_true")
    parser.add_argument("--skip_t2_question",     action="store_true")
    parser.add_argument("--skip_t2_words",      action="store_true")
    parser.add_argument("--skip_t2_pre_words",    action="store_true")
    parser.add_argument("--skip_t2_backchannel",  action="store_true")
    parser.add_argument("--skip_t3_backchannel",  action="store_true")
    parser.add_argument("--skip_t3_pauses",       action="store_true")
    parser.add_argument("--skip_t3_duration",   action="store_true")
    parser.add_argument("--skip_t4_words",      action="store_true")

    # ── GPT validity criteria (forwarded to extractor) ────────────────────────
    parser.add_argument("--skip_gpt_t1_complete",       action="store_true")
    parser.add_argument("--skip_gpt_t1_self_contained", action="store_true")
    parser.add_argument("--skip_gpt_t2_responds",           action="store_true")
    parser.add_argument("--skip_gpt_t3_floor_intent",       action="store_true")
    parser.add_argument("--skip_gpt_t3_pragmatic_complete", action="store_true")
    parser.add_argument("--skip_gpt_t3_mid_flow",           action="store_true")
    parser.add_argument("--skip_gpt_t4_meaningful",         action="store_true")

    args = parser.parse_args()
    CANDOR_ROOT = args.candor_root.expanduser().resolve()

    if args.no_gpt and args.results_dir == DEFAULT_RESULTS_DIR:
        args.results_dir = Path(str(DEFAULT_RESULTS_DIR) + "_no_gpt")

    # Build the list of extra flags to forward to each extractor subprocess
    extra_args: list[str] = []
    bool_flags = [
        "skip_abab", "skip_t2_overlap_t1", "skip_t3_overlap", "skip_t4_overlap_t3",
        "skip_t1_sent_final", "skip_t1_words", "skip_t1_pauses", "skip_t1_backchannel",
        "skip_t2_question", "skip_t2_words", "skip_t2_pre_words", "skip_t2_backchannel",
        "skip_t3_backchannel", "skip_t3_pauses", "skip_t3_duration", "skip_t4_words",
        "skip_gpt_t1_complete", "skip_gpt_t1_self_contained",
        "skip_gpt_t2_responds",
        "skip_gpt_t3_floor_intent", "skip_gpt_t3_pragmatic_complete", "skip_gpt_t3_mid_flow",
        "skip_gpt_t4_meaningful",
    ]
    for flag in bool_flags:
        if getattr(args, flag, False):
            extra_args.append(f"--{flag}")
    if args.min_words_filter is not None:
        extra_args += ["--min_words_filter", str(args.min_words_filter)]
    extra_args += ["--min_t3_overlap",    str(args.min_t3_overlap)]
    extra_args += ["--max_t2_t1_overlap", str(args.max_t2_t1_overlap)]
    extra_args += ["--max_t4_t3_overlap", str(args.max_t4_t3_overlap)]

    # Derive a unique per-conversation subdirectory from the results_dir name so
    # that concurrent runs with different settings never overwrite each other's files.
    rname = args.results_dir.name
    conv_subdir = rname.replace("interruptions_", "interruption_candidates_", 1)
    if not conv_subdir.startswith("interruption_candidates_"):
        conv_subdir = f"interruption_candidates_{rname}"
    extra_args += ["--conv_subdir", conv_subdir]

    all_convs  = find_conversations(args.parts)
    todo_convs = all_convs if args.force else [
        c for c in all_convs if not is_done(c, args.no_gpt, conv_subdir)]
    done_convs = [c for c in all_convs if is_done(c, args.no_gpt, conv_subdir)]

    print(f"Found {len(all_convs)} conversations total")
    print(f"  Already processed : {len(done_convs)}")
    print(f"  To process        : {len(todo_convs)}")
    print(f"  Workers           : {args.workers}")
    print(f"  Mode              : {'timing only (--no_gpt)' if args.no_gpt else 'full (with GPT)'}")
    print(f"  Target/type       : {args.target}")
    print(f"  Results dir       : {args.results_dir}")
    print(f"  Conv subdir       : {conv_subdir}")
    print(f"  min_t3_overlap    : {args.min_t3_overlap}s")
    print(f"  max_t2_t1_overlap : {args.max_t2_t1_overlap}s")
    print(f"  max_t4_t3_overlap : {args.max_t4_t3_overlap}s")
    if extra_args:
        print(f"  Extra flags       : {' '.join(extra_args)}")

    if args.dry_run:
        for c in todo_convs:
            print(f"  {c}")
        return

    if not todo_convs:
        print("\nNothing to do. Use --force to reprocess.")
        aggregate_results(args.results_dir, all_convs, args.no_gpt, args.target, conv_subdir)
        return

    print(f"\nStarting batch...\n{'─'*60}")
    batch_start   = time.time()
    results       = []
    n_done        = 0
    n_failed      = 0
    total_valid   = 0
    total_timing  = 0
    quota_abort   = False
    global_timing: dict[str, int] = {}
    global_gpt:   dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, c, args.results_dir, args.no_gpt, extra_args): c
            for c in todo_convs
        }
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            n_done += 1

            if res.get("quota_hit"):
                quota_abort = True
                for f in futures:
                    f.cancel()
                print(
                    f"\n{'!'*60}\n"
                    f"  QUOTA EXCEEDED — aborting batch.\n"
                    f"  Failed at: {Path(res['conv_dir']).name}\n"
                    f"  Completed so far: {n_done}/{len(todo_convs)}\n"
                    f"{'!'*60}"
                )
                break

            status = "OK " if res["success"] else "ERR"
            print(
                f"[{n_done:3d}/{len(todo_convs)}] {status}  "
                f"{Path(res['conv_dir']).name}  "
                f"timing={res['timing_found']}  valid={res['valid_found']}  "
                f"{res['elapsed']}s"
            )
            if not res["success"]:
                n_failed += 1
                print(f"         STDERR: {res['stderr']}")
            else:
                total_timing += res["timing_found"]
                total_valid  += res["valid_found"]
                for k, v in res.get("timing_stats", {}).items():
                    global_timing[k] = global_timing.get(k, 0) + v
                for k, v in res.get("gpt_stats", {}).items():
                    if isinstance(v, int):
                        global_gpt[k] = global_gpt.get(k, 0) + v

    elapsed_total = round(time.time() - batch_start, 1)
    print(f"\n{'─'*60}")
    if quota_abort:
        print(f"Batch ABORTED (quota exceeded) after {elapsed_total}s")
    else:
        print(f"Batch complete in {elapsed_total}s")
        print(f"  Succeeded        : {n_done - n_failed}/{len(todo_convs)}")
        print(f"  Failed           : {n_failed}")
        print(f"  Total timing hits: {total_timing}")
        if not args.no_gpt:
            print(f"  Total valid (GPT): {total_valid}")

    if global_timing:
        total_checked = global_timing.get("windows_checked", 1) or 1
        print(f"\n── Timing filter breakdown (all {n_done - n_failed} conversations) ──")
        for key, label in TIMING_LABELS.items():
            val = global_timing.get(key, 0)
            if key == "windows_checked":
                print(f"  {label}: {val}")
            else:
                pct = 100.0 * val / total_checked
                print(f"  {label}: {val} ({pct:.1f}%)")

    if not args.no_gpt and global_gpt:
        judged = global_gpt.get("total_judged", 1) or 1
        sent   = global_gpt.get("total_sent", 0)
        print(f"\n── GPT filter breakdown (all {n_done - n_failed} conversations) ──")
        print(f"  Sent to GPT          : {sent}")
        print(f"  Judged               : {judged}")
        print(f"  Valid after GPT      : {total_valid}")
        print(f"  Acceptance rate      : {100.0 * total_valid / judged:.1f}%")
        print(f"\n  Rejection criteria (% of judged):")
        gpt_criteria = [
            ("turn1_complete = false",         "turn1_complete_false",        "skip_gpt_t1_complete"),
            ("turn1_self_contained = false",   "turn1_self_contained_false",  "skip_gpt_t1_self_contained"),
            ("turn2_responds = false",         "turn2_responds_false",        "skip_gpt_t2_responds"),
            ("turn2_interrupted = false",      "turn2_interrupted_false",     None),
            ("t3_floor_intent = false",        "t3_floor_intent_false",       "skip_gpt_t3_floor_intent"),
            ("t3_pragmatic_complete = false",  "t3_pragmatic_complete_false", "skip_gpt_t3_pragmatic_complete"),
            ("t3_mid_flow_onset = false",      "t3_mid_flow_onset_false",     "skip_gpt_t3_mid_flow"),
            ("turn4_meaningful = false",       "turn4_meaningful_false",      "skip_gpt_t4_meaningful"),
        ]
        for label, key, skip_flag in gpt_criteria:
            val = global_gpt.get(key, 0)
            pct = 100.0 * val / judged
            skipped = bool(skip_flag and getattr(args, skip_flag, False))
            tag = "  [SKIPPED]" if skipped else ""
            print(f"    {label}: {val} ({pct:.1f}%){tag}")

    if n_failed:
        for r in results:
            if not r["success"]:
                print(f"  {r['conv_dir']}: {r['stderr']}")

    if not quota_abort:
        aggregate_results(args.results_dir, all_convs, args.no_gpt, args.target, conv_subdir)

        # ── Save filter stats for report generator ─────────────────────────────
        filter_stats = {
            "timing":      global_timing,
            "gpt":         global_gpt,
            "total_valid": total_valid,
            "n_convs":     n_done - n_failed,
        }
        (args.results_dir / "filter_stats.json").write_text(
            json.dumps(filter_stats, indent=2)
        )

        report_script = Path(__file__).parent / "generate_interruption_report.py"
        if report_script.exists():
            print("\nGenerating report...")
            report_cmd = [
                PYTHON, str(report_script),
                "--results_dir", str(args.results_dir),
                "--candor_root", str(CANDOR_ROOT),
                "--out", str(args.results_dir / "report.md"),
                "--conv_subdir", conv_subdir,
            ]
            if args.no_gpt:
                report_cmd.append("--no_gpt")
            # Forward any skipped GPT criteria so the report labels them correctly
            gpt_skip_flags = [
                "skip_gpt_t1_complete", "skip_gpt_t1_self_contained",
                "skip_gpt_t2_responds",
                "skip_gpt_t3_floor_intent", "skip_gpt_t3_pragmatic_complete", "skip_gpt_t3_mid_flow",
                "skip_gpt_t4_meaningful",
            ]
            for flag in gpt_skip_flags:
                if getattr(args, flag, False):
                    report_cmd.append(f"--{flag}")
            subprocess.run(report_cmd, check=False)


if __name__ == "__main__":
    main()
