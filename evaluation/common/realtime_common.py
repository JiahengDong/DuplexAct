#!/usr/bin/env python3
"""
evaluation/common/realtime_common.py
====================================
Shared infrastructure used by every closed-source realtime model client
(GPT-Realtime, Grok Voice, Qwen3.5-Omni-Flash-Realtime, Gemini 3.1 Flash
Live).

Provides:
  - EventLogger          : append-only JSONL log of every send/recv API event
  - SynchronizedRecorder  : real-time-paced recorder. Writes output.wav on the
                            SAME timeline as input.wav, and derives
                            `model_speech_segments` / `interruptions` directly
                            from the live stream (no VAD needed downstream).
  - RealtimeClientBase    : abstract per-sample session driver. Subclasses only
                            implement `_connect_and_stream(...)`; everything
                            else (recorder wiring, event logging, timing,
                            result-dict shape) is handled here so every model
                            produces byte-for-byte comparable output structure.
  - audio helpers         : load_and_resample()
  - result_dir_for()      : canonical action/l3_condition/model/sample_id path

Core timing principle (shared by all models, matches Full-Duplex-Bench):
    input.wav and output.wav share the SAME timeline, t=0 at stream start.
    output.wav has silence wherever the model was silent. input.wav already
    contains its own trailing silence "padding" (see each action's
    metadata.json: input_duration = content + padding_time), so the recorder
    target duration is simply metadata["input_duration"] -- no extra
    post-input grace window is needed.
"""

from __future__ import annotations

import abc
import asyncio
import json
import sys
import time
import traceback
import wave
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ── Shared constants ────────────────────────────────────────────────────────

REC_TICK_MS = 10   # SynchronizedRecorder loop granularity

# Post-input policy (design decision, supersedes the earlier drain_until_quiet
# heuristic from code-review F-01/F-11): the observation window is a hard,
# fixed cutoff at input_duration (== metadata.padding_end) for every provider,
# applied uniformly across all four L2 actions and both model classes. Once
# every input sample -- including the trailing padding -- has been sent, the
# receive task is cancelled immediately; we do not wait any further for new
# model audio, no matter whether/when the model would eventually respond.
# This matches how open-source batch-replay models are inherently bounded
# (they only ever process up to input_duration, since that's the file
# length), and gives every model the exact same, well-defined response
# window instead of a heuristic "wait until it looks done." Any response
# content that would have continued past this boundary is out of scope for
# scoring by design -- this is not truncation-as-bug, it's the definition of
# the window.


# ── Audio helpers ────────────────────────────────────────────────────────────

def load_and_resample(path: Path, target_sr: int) -> np.ndarray:
    """
    Load a WAV file, downmix to mono, resample to target_sr.
    Returns int16 numpy array.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != target_sr:
        g = gcd(int(target_sr), int(sr))
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return (data * 32767.0).clip(-32768, 32767).astype(np.int16)


# ── Replay-clock helpers (fixes code-review F-10) ───────────────────────────

async def sleep_until_deadline(loop: asyncio.AbstractEventLoop, deadline: float) -> None:
    """
    Sleep until an ABSOLUTE monotonic deadline, rather than a fixed relative
    duration. Every provider sender loop used to do
    `send(chunk); await asyncio.sleep(chunk_duration)`, which accumulates
    encoding/network/event-loop overhead into the replay schedule on every
    single frame (20ms frames over a 30-60s clip can drift by hundreds of ms).
    Callers instead track `deadline = stream_t0 + n * chunk_duration` for the
    n-th chunk and call this helper, so drift never compounds -- each sleep
    is computed relative to the fixed origin, not the previous sleep's actual
    wake time.
    """
    remaining = deadline - loop.time()
    if remaining > 0:
        await asyncio.sleep(remaining)


# ── Response-status normalization (code review R-06) ────────────────────────
#
# GPT, Qwen, and Grok all speak the OpenAI-Realtime-shaped `response.done`
# event with a `response.status` field, but the receive loops only used it
# (GPT) or didn't record it at all (Qwen/Grok) to flip `response_active`
# back to False -- a terminal `failed` or `incomplete` response was silently
# accepted as if it were an ordinary completed turn, and `cancelled` was
# never distinguished from a genuine client-recorded barge-in vs. some other,
# unexpected server-side cancellation.

def classify_terminal_response_status(
    status: Optional[str], expected_cancel: bool,
) -> tuple[str, bool]:
    """
    Normalize a provider's terminal `response.done` status into one of
    "completed", "expected_cancelled", "unexpected_cancelled", "failed",
    "incomplete", or "unknown", and say whether the caller should record it
    as a provider error (which ultimately fails the sample) rather than
    silently accepting it as ordinary model behavior.

    `expected_cancel` must be True when the cancellation is explained by
    barge-in -- either the client already called `recorder.interrupt()`
    after `speech_started` during an active response, or the provider
    reported `status_details.reason == "turn_detected"` (see
    `is_expected_cancel()`). That is legitimate full-duplex behavior, not
    a failure. Any other non-"completed" status (an unexplained
    "cancelled", "failed", or "incomplete") is not expected under this
    benchmark's protocol and should fail the sample instead of being
    accepted as valid output.
    """
    if status == "completed":
        return "completed", False
    if status == "cancelled":
        if expected_cancel:
            return "expected_cancelled", False
        return "unexpected_cancelled", True
    if status == "failed":
        return "failed", True
    if status == "incomplete":
        return "incomplete", True
    return (status or "unknown"), True


def is_expected_cancel(expecting_cancel: bool, status: Optional[str],
                       status_details=None) -> bool:
    """
    Whether a `response.done` cancel is explained by user barge-in.

    Providers do not emit `speech_started` and `response.done(cancelled)`
    in a fixed order. GPT typically sends `speech_started` first (the
    client sets `expecting_cancel`). Qwen's semantic_vad often emits
    `cancelled` + `reason=turn_detected` first, then `speech_started`
    ~1ms later -- by then `response_active` is already False, so the
    speech_started handler never arms `expecting_cancel`. Treating
    `turn_detected` as expected matches the simulator clients and
    avoids failing samples for a pipeline event-order mismatch.
    """
    if expecting_cancel:
        return True
    if status != "cancelled":
        return False
    details = status_details or {}
    reason = details.get("reason") if isinstance(details, dict) else None
    return reason == "turn_detected"


# ── Event logging ────────────────────────────────────────────────────────────

class EventLogger:
    """
    Append-only JSONL log of every send/recv event during a realtime session.
    Timestamps are wall-clock seconds relative to a single, fixed t=0: the
    moment input streaming actually begins (see RealtimeClientBase.
    start_streaming), matching the SynchronizedRecorder's own timeline so the
    two files can be cross-referenced directly.

    Events logged BEFORE t=0 is established (connection handshake,
    session.update, etc.) are buffered and retroactively re-timed once
    `finalize_start_time()` is called, so every line in the file ends up on
    the same timeline (handshake events naturally get negative "t").

    Raw audio bytes are never inlined (only byte counts), to keep the log
    small and text-diffable -- the audio itself lives in output.wav.
    """

    def __init__(self, path: Path, start_time: float):
        self.start_time = start_time   # provisional, until finalize_start_time()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w")
        self._pending: list[tuple[float, dict]] = []
        self._finalized = False

    def log(self, direction: str, etype: str, data: Optional[dict] = None,
            n_bytes: Optional[int] = None) -> None:
        now = time.monotonic()
        rec: dict = {"dir": direction, "type": etype}   # "dir": "send" | "recv"
        if n_bytes is not None:
            rec["bytes"] = n_bytes
        if data is not None:
            rec["data"] = data
        if self._finalized:
            self._write(now, rec)
        else:
            self._pending.append((now, rec))

    def finalize_start_time(self, start_time: float) -> None:
        """
        Establishes the true t=0. Retroactively times every buffered event
        relative to it, then switches to writing events immediately as logged.
        """
        self.start_time = start_time
        self._finalized = True
        for now, rec in self._pending:
            self._write(now, rec)
        self._pending = []

    def _write(self, now: float, rec: dict) -> None:
        ordered = {"t": round(now - self.start_time, 4), **rec}
        self._fh.write(json.dumps(ordered) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._finalized:
            self.finalize_start_time(self.start_time)
        if not self._fh.closed:
            self._fh.close()


# ── Synchronized recorder ────────────────────────────────────────────────────

class SynchronizedRecorder:
    """
    Time-synchronized audio recorder shared by all model clients.

    Writes `outfile` directly (streaming, not buffered in memory) at a pace
    matching real elapsed wall-clock time, so output.wav's own timeline is a
    faithful recording of when the model actually spoke -- independent of API
    network jitter or batch-vs-realtime chunk arrival.

    Ground-truth instrumentation (no VAD needed downstream):
      - self.segments      : list[(start_s, end_s)] of contiguous audio the
                              recorder actually wrote (i.e. the model's real
                              speech segments on the output.wav timeline)
      - self.interruptions : list[{"at_s": float, "reason": str}] logged
                              whenever interrupt() is called

    interrupt() semantics: called by a model client the moment the API signals
    that the model's in-progress response was cancelled/truncated (e.g.
    because the ongoing input audio triggered a server-side barge-in). This
    clears any audio chunks already queued-but-not-yet-written (audio that, in
    the true real-time signal, the model never got to finish "saying") and
    mutes the recorder until new audio arrives. It never touches input.wav --
    the replayed input keeps streaming on its own fixed schedule regardless.
    """

    def __init__(self, out_sr: int, target_sec: float, outfile: Path):
        self.out_sr = out_sr
        self.target_samples = int(round(target_sec * out_sr))
        self.outfile = Path(outfile)
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()

        self.tick_samples = max(1, out_sr * REC_TICK_MS // 1000)
        self._silence = np.zeros(self.tick_samples, dtype=np.int16)

        self.count = 0            # samples written so far
        self.muted = False
        self.running = True

        # R-03: response-identity gating. add() drops any packet whose
        # response_id is already recorded here as cancelled, instead of
        # relying solely on `muted` -- a straggler delta that was already
        # in flight before the provider's own cancellation/interrupt took
        # effect must not resume playback of audio the interruption was
        # meant to suppress. See add()/interrupt() docstrings.
        self.cancelled_response_ids: set = set()
        self.bytes_received = 0    # every add() call, before any gating
        self.bytes_accepted = 0    # queued for playback
        self.bytes_dropped  = 0    # dropped: belonged to a cancelled response
        self.bytes_flushed  = 0    # queued, then discarded by interrupt()

        self.segments: list[tuple[float, float]] = []
        self.interruptions: list[dict] = []
        self._open_segment_start: Optional[float] = None
        self._manual_segments = False   # see set_speaking()

        self.audio_bytes_written = 0
        self.silence_samples_written = 0
        self._stop_requested_at: Optional[float] = None
        self.abort = False   # see stop(abort=...)
        # Set externally (by RealtimeClientBase.start_streaming, see
        # run_sample()) to the EXACT same monotonic value used as t=0 for the
        # event log and for each client's send-deadline scheduling, so all
        # three timelines share one literal clock read instead of three
        # independently-measured "approximately t=0" instants (each of which
        # individually is off by at most a fraction of a millisecond, but
        # there's no reason to accept even that when the authoritative value
        # is one attribute-set away -- see the "how is Gemini's t=0 handled"
        # discussion in git history). If left unset, run() falls back to
        # measuring its own origin at task-start time.
        self.stream_origin: Optional[float] = None
        # Opt-in (Bayling): vocoder chunks arrive slower than realtime. The
        # default F-05 / post-target pad writes that GPU delay as silence
        # inside an open assistant turn, so output.wav is chopped and ASR
        # for Subsequent fails. When True, one open speaking segment is
        # written chunk-to-chunk; silence padding resumes after set_speaking
        # (False). Immediate onset stays the set_speaking(True) timestamp.
        self.contiguous_while_speaking = False

    async def add(self, pcm: bytes, response_id: Optional[str] = None) -> None:
        """
        Feed model audio into the recorder. Un-mutes if muted.

        response_id (R-03): identifies which provider response this packet
        belongs to (e.g. OpenAI/Qwen/Grok's `response_id` field on every
        audio-delta event). If it names a response already recorded as
        cancelled via `interrupt(response_id=...)`, the packet is dropped
        instead of queued -- a delta that was already in flight before the
        provider's cancellation took effect must not resume audio the
        interruption was supposed to suppress. Pass None (default) for
        events/providers with no response identity available; such packets
        are always accepted, matching the pre-fix behavior -- callers that
        CAN supply an id should always do so.
        """
        self.bytes_received += len(pcm)
        if response_id is not None and response_id in self.cancelled_response_ids:
            self.bytes_dropped += len(pcm)
            return
        self.bytes_accepted += len(pcm)
        if self.muted:
            self.muted = False
        await self.queue.put(pcm)

    def set_speaking(self, speaking: bool, at_s: Optional[float] = None) -> None:
        """
        Explicit segment-boundary control, for native full-duplex models
        (e.g. Moshi) that continuously emit an audio signal with no literal
        digital silence -- for these, `model_speech_segments` cannot be
        derived from audio presence/absence in the queue (that logic is
        what the automatic per-tick tracking in `run()` does, and is
        correct for turn-based APIs where "no audio queued" really does
        mean "not talking"). Call this instead, driven by whatever
        model-native signal indicates "is currently producing a word"
        (e.g. Moshi's inner-monologue text-token stream).

        Once called, this recorder permanently switches to manual mode:
        the automatic open/close tracking in `run()` is disabled for the
        rest of this recorder's lifetime, so the two mechanisms never
        fight over `segments`/`_open_segment_start`. `add()` keeps writing
        whatever audio arrives regardless of speaking state -- output.wav
        still faithfully contains the model's full continuous signal.

        at_s: timestamp on the recorder's own timeline (seconds since
        stream start). Defaults to "now" (self.count / self.out_sr) if
        the caller doesn't have a more precise value on hand (e.g. from
        its own event-loop clock).
        """
        self._manual_segments = True
        now = at_s if at_s is not None else self.count / self.out_sr
        if speaking:
            if self._open_segment_start is None:
                self._open_segment_start = now
        else:
            if self._open_segment_start is not None and now > self._open_segment_start:
                self.segments.append((round(self._open_segment_start, 4), round(now, 4)))
            self._open_segment_start = None

    def _hold_contiguous_speech(self) -> bool:
        return bool(self.contiguous_while_speaking and self._open_segment_start is not None)

    def current_speaking_duration(self, at_s: Optional[float] = None) -> float:
        """
        Added for evaluation/simulator_replay/'s live trigger controller (stop,
        hold_floor, interrupt_user, respond_and_recover): returns how many
        seconds the model has been CONTINUOUSLY speaking as of `at_s`
        (defaults to "now", self.count / self.out_sr), i.e. time since the
        currently-open segment (`_open_segment_start`) began. Returns 0.0 if
        the model is not currently speaking (no open segment). Works
        identically whether segments are tracked automatically (audio
        presence, turn-based APIs) or manually via set_speaking() (Moshi-
        style continuous models) -- both modes maintain `_open_segment_start`
        the same way, so no caller-visible distinction is needed here.
        """
        if self._open_segment_start is None:
            return 0.0
        now = at_s if at_s is not None else self.count / self.out_sr
        return max(0.0, now - self._open_segment_start)

    def played_speech_s(self) -> float:
        """
        Cumulative seconds of MODEL AUDIO actually written to output.wav so
        far -- i.e. how much of the model's speech the simulated listener has
        actually heard. Excludes silence padding, and excludes audio that was
        queued but dropped by interrupt() (the listener never heard that).

        Distinct from `count / out_sr` (that is wall-clock position, silence
        included) and from `current_speaking_duration()` (that is the length
        of the currently-open segment only).

        Added for the simulator trigger controller's ear-aligned transcript
        release: the provider's text stream runs 1.9x-40x faster than its own
        speech (measured per response over respond_and_recover: Gemini 1.9x,
        GPT 5.6x, Grok 6.8x, Qwen 39.6x), and on top of that the API delivers
        audio 4-5.5x faster than realtime, so `transcript_so_far` as received
        off the socket describes a point seconds ahead of the listener's ear.
        The controller converts this value into a character budget instead.
        """
        return self.audio_bytes_written / 2.0 / self.out_sr

    def queued_unplayed_s(self) -> float:
        """
        Seconds of model audio accepted into the queue but not yet written --
        the backlog the listener has still to hear. Used to work out when an
        item's playback will actually begin (its predecessor may still be
        draining) so `conversation.item.truncate`'s audio_end_ms can be
        expressed as played-ms-within-that-item.
        """
        pending = self.bytes_accepted - self.bytes_flushed - self.audio_bytes_written
        return max(0.0, pending / 2.0 / self.out_sr)

    def speaking_duration_between(self, t0: float, t1: float) -> float:
        """
        Seconds of model speech overlapping [t0, t1], including a still-open
        segment. Used by the simulator trigger controller to count speech
        AFTER lead-in (t0 = lead_in_end) without crediting barge-in that
        happened while the user was still talking.
        """
        if t1 <= t0:
            return 0.0
        total = 0.0
        for start, end in self.segments:
            total += max(0.0, min(end, t1) - max(start, t0))
        if self._open_segment_start is not None:
            total += max(0.0, t1 - max(self._open_segment_start, t0))
        return total

    def last_speech_end_after(self, t0: float, now: float) -> Optional[float]:
        """
        End timestamp of the latest *closed* segment that overlaps [t0, now].
        None if the model is currently speaking, or if no post-t0 speech has
        closed yet. Pair with current_speaking_duration() > 0 to distinguish
        "still talking" from "never talked after t0".
        """
        if self._open_segment_start is not None:
            return None
        last = None
        for start, end in self.segments:
            if end > t0 and start < now:
                last = end if last is None else max(last, end)
        return last

    def first_speech_start_after(self, t0: float, now: float) -> Optional[float]:
        """
        Start of the earliest model-speech that overlaps [t0, now], clipped
        to t0 (so a turn that began during lead-in counts as starting at
        lead-in end). None if there has been no post-t0 speech yet.
        """
        first = None
        def consider(start: float, end: float) -> None:
            nonlocal first
            if end <= t0 or start >= now:
                return
            clipped = max(start, t0)
            first = clipped if first is None else min(first, clipped)
        for start, end in self.segments:
            consider(start, end)
        if self._open_segment_start is not None:
            consider(self._open_segment_start, now)
        return first

    def interrupt(self, reason: str = "server_barge_in",
                  response_id: Optional[str] = None) -> None:
        """
        Drop any queued-but-unwritten audio and mute until new audio arrives.

        response_id (R-03): if the provider tells us which response was
        just cancelled, record it so that any later add() call carrying
        this same id -- a straggler delta that was already in flight
        before the cancellation took effect -- is dropped instead of
        silently resuming playback of audio the interruption was supposed
        to suppress. `cancelled_response_ids` accumulates across the whole
        sample (not just until the next add()) since a stale delta can in
        principle arrive arbitrarily late relative to the cancellation.
        """
        while not self.queue.empty():
            try:
                self.bytes_flushed += len(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        self.muted = True
        if response_id is not None:
            self.cancelled_response_ids.add(response_id)
        self._close_open_segment()
        self.interruptions.append({
            "at_s":       round(self.count / self.out_sr, 4),
            "reason":     reason,
            "response_id": response_id,
        })

    def stop(self, abort: bool = False) -> None:
        """
        abort=True (code review R-08): the caller is already unwinding due
        to an external cancellation/timeout, not a normal end-of-sample
        return. run() will drop any still-queued audio and pad straight to
        target_samples in one instantaneous write instead of the usual
        real-time-paced drain -- appropriate here because the sample is
        already being recorded as an error/timeout, so faithfully replaying
        remaining silence in real time would only make teardown itself take
        up to another target_samples worth of wall-clock time.
        """
        self.running = False
        self.abort = abort
        self._stop_requested_at = asyncio.get_event_loop().time()

    def _close_open_segment(self) -> None:
        if self._open_segment_start is not None:
            end_s = self.count / self.out_sr
            if end_s > self._open_segment_start:
                self.segments.append((round(self._open_segment_start, 4), round(end_s, 4)))
            self._open_segment_start = None

    async def run(self) -> None:
        """
        Fixes applied vs. the original implementation (see
        realtime_direct_replay_code_review.pdf F-05/F-06/F-07):

        F-05 (time compression): once `count` first reaches `target_samples`
        via silence padding, real audio arriving later used to be written
        immediately at position `target_samples` regardless of how much
        additional wall-clock time had actually elapsed -- silently deleting
        that gap from the timeline. We now track wall-clock time with a
        monotonic clock (`_expected_count()`) independent of `count`, and
        before writing any dequeued audio, first fill in whatever silence gap
        has opened up between `count` and "samples that should exist by now"
        so the audio lands at its true elapsed-time offset. This can make
        output.wav slightly LONGER than target_samples when a response
        genuinely continues past the nominal input_duration boundary --
        that's intentional (see module docstring: benchmark scoring already
        windows on metadata timestamps, not on raw file length), not a
        regression.

        F-06 (drop-on-stop): the loop used to exit as soon as
        `not self.running and count >= target_samples`, even if `self.queue`
        still held genuine received-but-unwritten model audio. It now also
        requires the queue to be empty before exiting (bounded by
        `_HARD_DRAIN_CAP_S` after stop() as a defensive fallback against any
        unforeseen producer bug, so this can never hang a batch job
        indefinitely).

        F-07 (chunk write-ahead / coarse interrupt timing): a whole received
        PCM chunk used to be written and slept-through in one shot, so
        `interrupt()` (called concurrently from the receive loop) could only
        take effect at whole-chunk boundaries -- up to the size of a single
        provider delta, which is coarser than our own REC_TICK_MS. Each
        chunk is now written in REC_TICK_MS-sized sub-pieces, checking
        `self.muted` between each one, so a barge-in signal truncates
        playback within one tick instead of only between deltas.
        """
        self.outfile.parent.mkdir(parents=True, exist_ok=True)
        wf = wave.open(str(self.outfile), "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.out_sr)

        loop = asyncio.get_event_loop()
        # Prefer the externally-supplied origin (the literal same clock read
        # RealtimeClientBase.start_streaming() used for the event log and
        # that each client's sender uses for deadline pacing) over measuring
        # our own -- see self.stream_origin's docstring in __init__.
        stream_t0 = self.stream_origin if self.stream_origin is not None else loop.time()
        _HARD_DRAIN_CAP_S = 10.0   # safety valve, see F-06 docstring above

        def expected_count() -> int:
            elapsed = loop.time() - stream_t0
            return max(self.count, int(round(elapsed * self.out_sr)))

        try:
            while True:
                if self.abort:
                    break   # R-08: skip real-time drain, pad instantly below
                queue_drained = self.queue.empty()
                # Exit as soon as the client stops the session and the
                # queue is empty. Do not require count >= target_samples:
                # simulator Stop/Hold-floor keep the socket open well past
                # lead-in (cue + POST_CUE_OBSERVATION_S). The final pad
                # below stretches the file to elapsed wall-clock so the
                # cue tail is on disk even if the model already went silent.
                if not self.running and queue_drained:
                    break
                if (self._stop_requested_at is not None
                        and loop.time() - self._stop_requested_at > _HARD_DRAIN_CAP_S):
                    break   # defensive fallback only -- see F-06 docstring

                if not self.muted and not self.queue.empty():
                    pcm = await self.queue.get()
                    smp = np.frombuffer(pcm, dtype=np.int16)

                    # Catch up on any real-time silence gap BEFORE writing
                    # this chunk, so it lands at its true elapsed-time
                    # offset instead of being compressed against `count`
                    # (F-05). Only relevant once we're past target_samples;
                    # inside normal playback expected_count()==count already.
                    # Skip while contiguous_while_speaking: the delay is
                    # vocoder lag, not a real pause in the utterance.
                    if not self._hold_contiguous_speech():
                        gap = expected_count() - self.count
                        if gap > 0:
                            wf.writeframes(np.zeros(gap, dtype=np.int16).tobytes())
                            self.silence_samples_written += gap
                            self.count += gap

                    if not self._manual_segments and self._open_segment_start is None:
                        self._open_segment_start = self.count / self.out_sr

                    # Genuine model audio is NEVER truncated, even once count
                    # has already reached target_samples -- matches
                    # BatchReplayClientBase's "pad up to input_duration if
                    # short, never truncate" contract (see its run_sample()).
                    # Without this, any model whose reply legitimately starts
                    # only once real wall-clock time has already carried the
                    # recorder's silence padding past target_samples (e.g.
                    # VITA-1.5, which has no incremental input encoder and
                    # only starts generating after input_duration has fully
                    # elapsed -- see infer_vita15.py's module docstring)
                    # would have every one of its real audio chunks silently
                    # dequeued-and-discarded here -- discovered via VITA-1.5's
                    # output.wav being all-zero digital silence despite dozens
                    # of confirmed successful TTS decode passes in its job
                    # log (see vita-1.5.md).
                    #
                    # Written in REC_TICK-sized sub-pieces (F-07) so a
                    # concurrent interrupt() can truncate mid-chunk instead
                    # of only between whole received deltas.
                    pos = 0
                    while pos < len(smp):
                        if self.muted:
                            break
                        n = min(self.tick_samples, len(smp) - pos)
                        piece = smp[pos:pos + n]
                        wf.writeframes(piece.tobytes())
                        self.count += n
                        self.audio_bytes_written += n * 2
                        pos += n
                        await asyncio.sleep(n / self.out_sr)
                    continue

                if not self._manual_segments:
                    self._close_open_segment()

                if self._hold_contiguous_speech():
                    # Wait for the next vocoder chunk without inserting
                    # lead-in or post-target silence into this utterance.
                    await asyncio.sleep(REC_TICK_MS / 1000)
                    continue

                if self.count >= self.target_samples:
                    # Past the nominal input/lead-in length. Keep writing
                    # wall-clock silence while the session is still live so
                    # Stop Subsequent can see the rest of the cue (model
                    # yielded, cue still playing). If we only sleep here,
                    # output.wav ends at the last model sample -- Gemini
                    # Stop cut 190/200 files before t_cue_end, and
                    # never_stopped was a file-length artifact.
                    # Later model audio still lands at true elapsed time
                    # because count tracks the silence we just wrote (F-05).
                    if self.running:
                        gap = expected_count() - self.count
                        if gap > 0:
                            n = min(self.tick_samples, gap)
                            wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())
                            self.silence_samples_written += n
                            self.count += n
                            await asyncio.sleep(n / self.out_sr)
                        else:
                            await asyncio.sleep(REC_TICK_MS / 1000)
                    else:
                        await asyncio.sleep(REC_TICK_MS / 1000)
                    continue

                remain = self.target_samples - self.count
                n = min(self.tick_samples, remain)
                wf.writeframes(self._silence[:n].tobytes())
                self.count += n
                self.silence_samples_written += n
                await asyncio.sleep(n / self.out_sr)

            self._close_open_segment()
            elapsed_samples = int(round((loop.time() - stream_t0) * self.out_sr))
            if self.abort:
                pad_to = self.target_samples
            else:
                # Cover the real session, not just lead-in. Simulator Stop
                # Subsequent is "quiet until t_cue_end"; that tail only
                # exists on disk if we pad to wall-clock elapsed.
                pad_to = max(self.target_samples, elapsed_samples, self.count)
            if self.count < pad_to:
                remaining = pad_to - self.count
                wf.writeframes(np.zeros(remaining, dtype=np.int16).tobytes())
                self.silence_samples_written += remaining
                self.count = pad_to
        finally:
            wf.close()


# ── Post-hoc audio-energy segmentation (fixes lychee-fd's broken native
#    round-trace timestamps -- see evaluation/simulator_replay/MODEL_COMPATIBILITY.md's
#    "SEVERE finding" section for the full root-cause writeup) ──────────────

def recompute_speech_segments_from_audio(
    wav_path: Path,
    window_s: float = 0.05,
    threshold_rms: float = 0.01,
    merge_gap_s: float = 1.0,
    min_segment_s: float = 0.15,
) -> list[tuple[float, float]]:
    """
    Derives `model_speech_segments` directly from a finished output.wav's
    own audio content, instead of trusting a model client's native
    "speaking interval" event bookkeeping. Added specifically because
    lychee-fd's own native `round_trace["events"]` timestamps are computed
    from an INPUT-TIMELINE offset (`idx / INPUT_SR`) that silently diverges
    from real wall-clock arrival time once any single round's own
    generation call overruns its dispatch budget (confirmed: an ~11s
    divergence on a real production sample) -- see
    infer_lychee_fd.py for the full writeup. The actual PCM samples `add()`ed into
    SynchronizedRecorder are NEVER affected by this (they're written in
    true arrival order at true arrival time) -- only the derived event
    LABELS are wrong, so re-deriving them from the audio itself sidesteps
    the bug entirely rather than trying to patch the broken timestamp math.

    Method: windowed RMS envelope (int16 PCM normalized to [-1, 1], so
    `threshold_rms` is directly comparable across any output sample rate)
    thresholded into speech/silence per `window_s`-sized window, then two
    passes of real-signal cleanup:
      1. `merge_gap_s`: bridges short silence gaps BETWEEN speech windows
         -- needed because round-based/chunked incremental APIs (lychee-fd
         included) often deliver a continuous utterance's audio in bursts
         with brief (100s of ms) empty rounds interleaved even while the
         model has not actually paused speaking (confirmed on lychee-fd:
         alternating ~0-sample/~24000-sample rounds throughout one
         continuous ~13s reply) -- these are dispatch-granularity
         artifacts, not real pauses, and would otherwise fragment one true
         utterance into dozens of spurious micro-segments.
      2. `min_segment_s`: drops isolated speech-window blips too short to
         plausibly be real speech (encoder/decoder transients, not
         content) -- mirrors the same "don't score noise as speech" intent
         `direct_replay`'s automatic (audio-presence) segmentation already
         gets for free from providers whose deltas arrive in
         speech-length-sized chunks; a coarse windowed RMS scan needs this
         explicit guard since individual windows are much shorter.

    Defaults chosen empirically against actual lychee-fd output.wav files
    (confirmed digital silence reads ~0.0 RMS, real speech content reads
    ~0.1 RMS on the same [-1, 1] scale -- see the incident writeup) --
    `threshold_rms=0.01` sits comfortably in between (~20dB below real
    speech, ~infinitely above true digital silence). Model-agnostic: takes
    no dependency on any model's own native event/state signal, so this is
    safe to reuse for any model whose native segmentation signal is
    suspect, not just lychee-fd.
    """
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    pcm = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    win = max(1, int(round(window_s * sr)))
    n_windows = int(np.ceil(len(pcm) / win)) if len(pcm) else 0

    raw_segments: list[tuple[float, float]] = []
    open_start: Optional[float] = None
    for i in range(n_windows):
        seg = pcm[i * win:(i + 1) * win]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) else 0.0
        t = i * window_s
        if rms > threshold_rms:
            if open_start is None:
                open_start = t
        else:
            if open_start is not None:
                raw_segments.append((open_start, t))
                open_start = None
    if open_start is not None:
        raw_segments.append((open_start, n_windows * window_s))

    merged: list[list[float]] = []
    for s, e in raw_segments:
        if merged and s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(round(s, 4), round(e, 4)) for s, e in merged if e - s >= min_segment_s]


# ── Dynamic PCM source (evaluation/simulator_replay/ only) ─────────────────────────

class DynamicPCMSource:
    """
    Growable, silence-padded PCM byte source for simulator-driven samples
    (stop / hold_floor / interrupt_user / respond_and_recover) where the
    total input timeline is NOT known upfront -- unlike direct_replay's
    single fixed `input_pcm` array, here a live LiveTriggerController
    decides mid-stream when to append the next clip (e.g. `cue.wav`) based
    on the model's own real-time speech behavior.

    Usage from a model client's sender loop (same per-chunk pacing pattern
    as every direct_replay client's `for i, off in enumerate(range(0,
    len(pcm_bytes_all), CHUNK_BYTES))`, just pulling from `source.pull(n)`
    instead of slicing a pre-built static array):

        i = 0
        while not source.is_done:
            chunk = source.pull(CHUNK_BYTES)
            await ws.send(...chunk...)
            await sleep_until_deadline(loop, send_t0 + (i + 1) * (CHUNK_MS / 1000))
            i += 1

    Semantics:
      - `append(pcm_bytes)` queues more real content (called once upfront
        with lead_in.wav's bytes, then again by the trigger controller with
        cue.wav's bytes once it decides "now").
      - `pull(n)` ALWAYS returns exactly n bytes immediately -- backed by
        real queued content when available, else digital silence. This is
        deliberate: "waiting for the trigger" must sound like silence to
        the model, not a paused/blocked stream, exactly matching how a real
        listener falls silent while thinking rather than freezing mid-audio.
      - `finish()` marks that no further content will ever be appended.
        `is_done` becomes True once the buffer is ALSO fully drained (i.e.
        pull() has returned every real byte, or the caller only ever needs
        to keep pulling silence in the meantime -- `is_done` only reflects
        "no real content left AND finished", callers that want a bounded
        post-clip silence tail should stop pulling on their own timer
        instead of relying on is_done to end the loop, since a finished-
        but-not-yet-fully-silent-tail source with no more real content
        still reports is_done=True as soon as the buffer empties).
    """

    def __init__(self, initial_pcm: bytes = b""):
        self._buffer = bytearray(initial_pcm)
        self._finished = False

    def append(self, pcm_bytes: bytes) -> None:
        self._buffer.extend(pcm_bytes)

    def truncate_pending(self) -> int:
        """
        Added for evaluation/simulator_replay's `interrupt_user` action (see
        interrupt_controller.py's own module docstring for the full
        rationale): unlike stop/hold_floor/respond_and_recover -- where the
        simulator injects a cue WHILE the model is speaking and never needs
        to un-queue anything it already committed to sending -- interrupt_user
        reverses the roles (the simulator is continuously "speaking" a
        pre-recorded user utterance, and it's the MODEL that may barge in).
        The moment the model's own onset is detected, a real interrupted
        user stops talking immediately rather than finishing their queued
        sentence -- so whatever tail of the current clip hasn't been pulled
        yet needs to be dropped, not delivered late after the model has
        already started talking over it. Returns the number of bytes
        dropped (for logging). Idempotent: calling this on an
        already-empty buffer is a no-op (returns 0). Does NOT set
        `_finished` -- callers still call `finish()` separately once they've
        appended whatever comes next (e.g. a branch reply clip), exactly
        like every other DynamicPCMSource caller already does.
        """
        n = len(self._buffer)
        self._buffer.clear()
        return n

    def finish(self) -> None:
        self._finished = True

    @property
    def is_done(self) -> bool:
        return self._finished and len(self._buffer) == 0

    def pull(self, n: int) -> bytes:
        if len(self._buffer) >= n:
            chunk = bytes(self._buffer[:n])
            del self._buffer[:n]
            return chunk
        # Partial or empty buffer: drain what's real, pad the rest with
        # silence. Never blocks.
        chunk = bytes(self._buffer)
        self._buffer.clear()
        return chunk + b"\x00" * (n - len(chunk))


# ── Result path helper ───────────────────────────────────────────────────────

def result_dir_for(results_root: Path, action: str, l3_condition: str,
                    model_slug: str, sample_id: str) -> Path:
    """
    Canonical inference_results path:
      <results_root>/<action>/<l3_condition>/<model_slug>/<sample_id>/
    """
    return results_root / action / l3_condition / model_slug / sample_id


# ── Native text → per-segment transcripts ────────────────────────────────────

def bucket_text_tokens_into_segments(text_tokens: list, segments: list) -> list[str]:
    """
    Bucket a flat list of (t, token_text) tuples -- as captured live by a
    client's own receive loop, on the same recorder clock `segments` are on --
    into each finalized speech segment.

    No separator between tokens on purpose: every model surveyed whose native
    text stream is word/subword-tokenized (Moshi, FLM-Audio, BayLing,
    MiniCPM-o, Raon) emits tokens that already carry their own leading space at
    real word boundaries, so joining with "" reproduces normal spacing while
    joining with " " double-spaces everything and glues punctuation awkwardly
    ("day ,  how"). A client whose tokens do NOT self-space needs its own
    `compute_model_speech_transcripts()` override rather than a change here.

    This is the same rule as the simulator's
    `stop_common.default_model_speech_transcripts()`, which is where it
    originated; direct replay cannot import from the simulator tree (the
    dependency runs the other way), so the implementation lives here in the
    shared lower layer. THE TWO MUST STAY IDENTICAL -- `run_metrics` scores
    both trees off this field, and a divergence would mean the same audio
    yields different text depending on which tree produced it. Fold
    stop_common's copy into a re-export of this function when no simulator job
    is mid-flight.
    """
    out = []
    for on, off in segments:
        words = [w for (t, w) in text_tokens if on <= t <= off]
        out.append("".join(words).strip())
    return out


# ── Realtime client base ─────────────────────────────────────────────────────

class RealtimeClientBase(abc.ABC):
    """
    One instance handles exactly one sample's full realtime session
    lifecycle: connect, stream input, receive + record output, log every
    event, and return a structured "inference" dict.

    Subclasses set the class attributes and implement `_connect_and_stream`;
    everything else (recorder wiring, event logging, timing, result shape) is
    identical across models so results are directly comparable.
    """

    MODEL_SLUG: str = "unset"    # directory-safe key, must match model_registry.py
    OUTPUT_SR: int = 24_000      # sample rate of audio returned by the model
    INPUT_SR: int = 24_000       # sample rate the API expects for input audio
    # See SynchronizedRecorder.contiguous_while_speaking. Bayling only.
    CONTIGUOUS_WHILE_SPEAKING: bool = False

    def __init__(self, api_key: str, model_name: str, system_prompt: str,
                 verbose: bool = False):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose

    async def run_sample(self, input_wav: Path, output_wav: Path,
                          events_path: Path, input_duration: float) -> dict:
        """
        Orchestrates one full sample: recorder + event logger wiring, calls
        `_connect_and_stream`, and returns the "inference" dict to be merged
        with instance metadata by the caller (run_inference.py).

        Timing reference: t=0 for BOTH the recorder (and therefore
        model_speech_segments/interruptions) and events.jsonl is the moment
        input streaming actually begins -- NOT when the connection is opened.
        Connection/session-setup handshake events are logged with negative
        timestamps. This matters because every sample's metadata.json
        (pause_start, bc_start, side_speech_start_s, ...) is relative to
        input.wav's own t=0, so recorder timestamps must align with that same
        origin regardless of how long a given API's handshake takes.
        """
        setup_t0 = time.monotonic()
        event_logger = EventLogger(events_path, setup_t0)
        recorder = SynchronizedRecorder(self.OUTPUT_SR, input_duration, output_wav)
        recorder.contiguous_while_speaking = bool(self.CONTIGUOUS_WHILE_SPEAKING)

        stream_start: dict = {"t": None}
        recorder_task_holder: dict = {"task": None}

        async def start_streaming() -> float:
            """
            Must be called by `_connect_and_stream` immediately before
            sending the first input audio chunk (and safe to call again on
            every later reconnect -- idempotent). Establishes t=0 for the
            recorder and the event log, and returns that exact monotonic
            value so callers can anchor their OWN send-deadline pacing
            (see sleep_until_deadline()) to the literal same clock read
            instead of taking a second, independent `loop.time()` sample a
            few instructions later -- which would technically be a different
            (if negligibly so) instant on the same clock. Recorder,
            event log, and sender pacing all end up sharing one number.
            """
            if stream_start["t"] is not None:
                return stream_start["t"]
            stream_start["t"] = time.monotonic()
            recorder.stream_origin = stream_start["t"]
            event_logger.finalize_start_time(stream_start["t"])
            recorder_task_holder["task"] = asyncio.create_task(recorder.run())
            return stream_start["t"]

        input_pcm = load_and_resample(input_wav, self.INPUT_SR)

        errors: list[str] = []
        aborted = False
        try:
            await self._connect_and_stream(input_pcm, input_duration, recorder,
                                            event_logger, start_streaming)
        except asyncio.CancelledError:
            # External cancellation (code review R-08) -- typically
            # run_inference.py's per-sample asyncio.wait_for timeout firing.
            # Must still re-raise (see below) so wait_for correctly observes
            # the cancellation and converts it to TimeoutError -- swallowing
            # it here would make the sample look like it finished normally
            # within the deadline. What we DO change is the finally block's
            # behavior: skip the real-time-paced sleep/padding entirely,
            # since this sample is already being torn down as a failure and
            # faithfully replaying its remaining silence in real time would
            # only make cleanup itself take up to another input_duration
            # seconds before the timeout is actually honored.
            aborted = True
            raise
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            tb = traceback.format_exc()
            if self.verbose:
                print(tb, file=sys.stderr)
            event_logger.log("recv", "client_error", {"message": str(e), "traceback": tb})
        finally:
            if recorder_task_holder["task"] is None:
                # Connection never got far enough to start streaming (e.g.
                # connect failed immediately) -- start now so output.wav is
                # still produced (all silence) and the batch doesn't stall.
                await start_streaming()
            if not aborted:
                elapsed = time.monotonic() - stream_start["t"]
                if elapsed < input_duration:
                    await asyncio.sleep(input_duration - elapsed)
            recorder.stop(abort=aborted)
            await recorder_task_holder["task"]
            event_logger.close()

        wallclock = time.monotonic() - stream_start["t"]
        return {
            "model":                 self.model_name,
            "input_duration_s":      round(input_duration, 4),
            "output_duration_s":     round(recorder.count / recorder.out_sr, 4),
            "model_speech_segments": [list(s) for s in recorder.segments],
            "model_speech_transcripts": self._model_speech_transcripts(recorder.segments),
            "interruptions":         recorder.interruptions,
            # R-03 telemetry: received == accepted + dropped. dropped > 0
            # means at least one late delta for an already-cancelled
            # response was correctly rejected instead of resuming playback.
            "audio_bytes_received":  recorder.bytes_received,
            "audio_bytes_accepted":  recorder.bytes_accepted,
            "audio_bytes_dropped":   recorder.bytes_dropped,
            "errors":                errors,
            "inference_wallclock_s": round(wallclock, 4),
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }

    def _model_speech_transcripts(self, segments: list) -> list[str]:
        """
        What the model itself said in each of its own speech segments, for the
        slugs `run_metrics.LIVE_TRANSCRIPT_MODELS` scores off native text
        instead of Whisper (BayLing and Raon, whose output.wav is >50% silence
        padding and defeats ASR).

        A client opts in by accumulating `self.text_tokens` as (t, text) on the
        recorder's clock, or by overriding
        `compute_model_speech_transcripts(segments)` when its native text needs
        different handling. Clients with no text signal at all get a list of
        "" -- the length always matches `segments`, never a short list or None,
        because `run_metrics.segment_texts()` decides whether native text is
        usable by checking `len(native) == len(segs)`.
        """
        fn = getattr(self, "compute_model_speech_transcripts", None)
        if fn is not None:
            return fn(segments)
        tokens = getattr(self, "text_tokens", None)
        if tokens:
            return bucket_text_tokens_into_segments(tokens, segments)
        return [""] * len(segments)

    @abc.abstractmethod
    async def _connect_and_stream(self, input_pcm: np.ndarray, input_duration: float,
                                   recorder: SynchronizedRecorder,
                                   event_logger: EventLogger,
                                   start_streaming) -> None:
        """
        Model-specific: open the realtime connection and complete any
        session-setup handshake, then call `await start_streaming()` exactly
        once, immediately before sending the first input audio chunk (this
        establishes t=0 for the recorder and event log). Then stream
        input_pcm in real time, feed received model audio into
        `await recorder.add(pcm_bytes)`, call
        `recorder.interrupt(reason=...)` the moment the server signals the
        model's response was cancelled/truncated by new input speech, and log
        every send/recv event via `event_logger.log(...)`.

        Must not run longer than necessary -- the base class already pads out
        to input_duration afterward if this returns early.
        """
        raise NotImplementedError
