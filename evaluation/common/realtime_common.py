#!/usr/bin/env python3
"""
Shared infrastructure for realtime model clients (GPT-Realtime, Grok Voice,
Qwen3.5-Omni-Flash-Realtime, Gemini 3.1 Flash Live, and streaming open models).

  - EventLogger          : append-only JSONL log of every send/recv API event
  - SynchronizedRecorder : real-time-paced recorder that writes output.wav on
                           the same timeline as input.wav and records
                           `model_speech_segments` / `interruptions` live
  - RealtimeClientBase   : per-sample session driver; subclasses implement
                           `_connect_and_stream(...)`
  - load_and_resample(), result_dir_for()

Timing: input.wav and output.wav share one timeline with t=0 at stream start;
output.wav is silent wherever the model was silent. input.wav already includes
its trailing padding (metadata input_duration = content + padding_time), so the
recorder target duration is metadata["input_duration"].
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

# Post-input policy: the observation window ends at input_duration
# (== metadata.padding_end) for every provider and action. Once all input,
# including trailing padding, has been sent, the receive task is cancelled.
# This matches batch-replay models, which only ever see input_duration of
# audio; anything the model would say after that is outside the scored window.


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


# ── Replay-clock helpers ─────────────────────────────────────────────────────

async def sleep_until_deadline(loop: asyncio.AbstractEventLoop, deadline: float) -> None:
    """
    Sleep until an absolute monotonic deadline. Sender loops pass
    `deadline = stream_t0 + n * chunk_duration` for the n-th chunk so that
    per-frame send overhead does not accumulate into replay drift.
    """
    remaining = deadline - loop.time()
    if remaining > 0:
        await asyncio.sleep(remaining)


# ── Response-status normalization ────────────────────────────────────────────
#
# GPT, Qwen, and Grok all emit an OpenAI-Realtime-shaped `response.done` event
# with a `response.status` field. A terminal `failed`/`incomplete` response,
# or a cancellation not explained by barge-in, should fail the sample rather
# than be accepted as an ordinary turn.

def classify_terminal_response_status(
    status: Optional[str], expected_cancel: bool,
) -> tuple[str, bool]:
    """
    Normalize a terminal `response.done` status into one of "completed",
    "expected_cancelled", "unexpected_cancelled", "failed", "incomplete", or
    "unknown", and return whether the caller should record it as a provider
    error (which fails the sample).

    `expected_cancel` is True when the cancellation is explained by barge-in:
    the client already called `recorder.interrupt()` after `speech_started`,
    or the provider reported `status_details.reason == "turn_detected"` (see
    `is_expected_cancel()`).
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
    in a fixed order. GPT usually sends `speech_started` first (the client
    sets `expecting_cancel`). Qwen's semantic_vad often emits `cancelled` +
    `reason=turn_detected` first, so `expecting_cancel` is never armed;
    `turn_detected` is therefore also treated as expected.
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
    Timestamps are seconds relative to the moment input streaming begins
    (RealtimeClientBase.start_streaming), the same t=0 as the recorder.

    Events logged before t=0 (handshake, session.update, ...) are buffered
    and re-timed by `finalize_start_time()`, so they get negative "t".
    Audio payloads are logged as byte counts only.
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
        Set t=0, write buffered events relative to it, and write later events
        immediately.
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

    Streams `outfile` to disk at real wall-clock pace, so output.wav records
    when the model actually spoke regardless of how fast chunks arrive.

      - self.segments      : list[(start_s, end_s)] of model audio written,
                             on the output.wav timeline
      - self.interruptions : list[{"at_s", "reason", "response_id"}], one per
                             interrupt() call

    interrupt() is called when the API signals that the in-progress response
    was cancelled (e.g. server-side barge-in). It drops queued-but-unwritten
    audio and mutes until new audio arrives; input streaming is unaffected.
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

        # add() drops packets whose response_id is listed here, so late
        # deltas from a cancelled response cannot resume playback.
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
        # Set by RealtimeClientBase.start_streaming to the same monotonic t=0
        # used by the event log and sender pacing. If unset, run() uses its
        # own start time.
        self.stream_origin: Optional[float] = None
        # Opt-in for models whose vocoder chunks arrive slower than realtime
        # (Bayling): while a set_speaking(True) segment is open, chunks are
        # written back-to-back instead of padding the generation delay with
        # silence, which would chop the utterance. Padding resumes after
        # set_speaking(False).
        self.contiguous_while_speaking = False

    async def add(self, pcm: bytes, response_id: Optional[str] = None) -> None:
        """
        Feed model audio into the recorder. Un-mutes if muted.

        response_id: the provider response this packet belongs to (e.g. the
        `response_id` on OpenAI/Qwen/Grok audio-delta events). Packets from a
        response already cancelled via `interrupt(response_id=...)` are
        dropped. Pass None when the provider has no response identity; such
        packets are always accepted.
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
        Explicit segment-boundary control for full-duplex models (e.g. Moshi)
        that emit audio continuously, so segments cannot be derived from
        whether audio is queued. Drive it from a model-native signal such as
        Moshi's inner-monologue text tokens.

        The first call switches the recorder to manual mode for its lifetime:
        automatic segment tracking in `run()` is disabled. `add()` still
        writes all audio regardless of speaking state.

        at_s: seconds since stream start; defaults to self.count / self.out_sr.
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
        Seconds the model has been continuously speaking as of `at_s`
        (default: now), i.e. time since the open segment began; 0.0 if no
        segment is open. Used by the simulator_replay trigger controller.
        Works in both automatic and set_speaking() modes.
        """
        if self._open_segment_start is None:
            return 0.0
        now = at_s if at_s is not None else self.count / self.out_sr
        return max(0.0, now - self._open_segment_start)

    def played_speech_s(self) -> float:
        """
        Cumulative seconds of model audio written to output.wav so far, i.e.
        how much the simulated listener has heard. Excludes silence padding
        and audio dropped by interrupt(). Unlike `count / out_sr` (wall-clock
        position) or `current_speaking_duration()` (open segment only).

        Providers stream text and audio well ahead of realtime playback, so
        the simulator trigger controller converts this value into a
        transcript character budget rather than trusting the received text.
        """
        return self.audio_bytes_written / 2.0 / self.out_sr

    def queued_unplayed_s(self) -> float:
        """
        Seconds of model audio queued but not yet written. Used to compute
        when an item's playback begins, so `conversation.item.truncate`'s
        audio_end_ms can be expressed as played ms within that item.
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

        response_id: the cancelled response, if known. Later add() calls
        carrying this id are dropped. `cancelled_response_ids` accumulates
        for the whole sample, since a stale delta can arrive arbitrarily late.
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
        abort=True: the caller is unwinding after cancellation/timeout. run()
        drops queued audio and pads to target_samples in one write instead of
        draining in real time, so teardown of a failed sample is immediate.
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
        Recorder loop.

        - Before writing dequeued audio, any silence gap between `count` and
          elapsed wall-clock time is filled, so audio lands at its true
          offset. output.wav may therefore run longer than target_samples;
          scoring windows on metadata timestamps, not file length.
        - After stop(), the loop exits only once the queue is empty, bounded
          by `_HARD_DRAIN_CAP_S`.
        - Chunks are written in REC_TICK_MS pieces, checking `self.muted`
          between pieces, so interrupt() truncates playback within one tick.
        """
        self.outfile.parent.mkdir(parents=True, exist_ok=True)
        wf = wave.open(str(self.outfile), "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.out_sr)

        loop = asyncio.get_event_loop()
        stream_t0 = self.stream_origin if self.stream_origin is not None else loop.time()
        _HARD_DRAIN_CAP_S = 10.0   # max drain time after stop()

        def expected_count() -> int:
            elapsed = loop.time() - stream_t0
            return max(self.count, int(round(elapsed * self.out_sr)))

        try:
            while True:
                if self.abort:
                    break   # skip real-time drain, pad instantly below
                queue_drained = self.queue.empty()
                # Exit once stopped and drained, without requiring
                # count >= target_samples: simulator Stop/Hold-floor keep the
                # session open past lead-in. The final pad below stretches
                # the file to elapsed wall-clock time.
                if not self.running and queue_drained:
                    break
                if (self._stop_requested_at is not None
                        and loop.time() - self._stop_requested_at > _HARD_DRAIN_CAP_S):
                    break

                if not self.muted and not self.queue.empty():
                    pcm = await self.queue.get()
                    smp = np.frombuffer(pcm, dtype=np.int16)

                    # Fill any wall-clock silence gap so this chunk lands at
                    # its true elapsed-time offset. Skipped while holding
                    # contiguous speech: that delay is vocoder lag, not a
                    # pause in the utterance.
                    if not self._hold_contiguous_speech():
                        gap = expected_count() - self.count
                        if gap > 0:
                            wf.writeframes(np.zeros(gap, dtype=np.int16).tobytes())
                            self.silence_samples_written += gap
                            self.count += gap

                    if not self._manual_segments and self._open_segment_start is None:
                        self._open_segment_start = self.count / self.out_sr

                    # Model audio is never truncated, even past target_samples
                    # (same contract as BatchReplayClientBase): some models,
                    # e.g. VITA-1.5, only start replying after input_duration.
                    # Written in REC_TICK-sized pieces so interrupt() can
                    # truncate mid-chunk.
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
                    # wall-clock silence while the session is live so
                    # output.wav covers the rest of the cue even if the model
                    # has gone quiet (Stop scoring needs audio up to
                    # t_cue_end).
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
                # Pad to wall-clock elapsed so the file covers the whole
                # session, not just lead-in.
                pad_to = max(self.target_samples, elapsed_samples, self.count)
            if self.count < pad_to:
                remaining = pad_to - self.count
                wf.writeframes(np.zeros(remaining, dtype=np.int16).tobytes())
                self.silence_samples_written += remaining
                self.count = pad_to
        finally:
            wf.close()


# ── Post-hoc audio-energy segmentation ───────────────────────────────────────

def recompute_speech_segments_from_audio(
    wav_path: Path,
    window_s: float = 0.05,
    threshold_rms: float = 0.01,
    merge_gap_s: float = 1.0,
    min_segment_s: float = 0.15,
) -> list[tuple[float, float]]:
    """
    Derive `model_speech_segments` from a finished output.wav's audio,
    for models whose native speaking-interval timestamps are unreliable
    (e.g. lychee-fd, whose round-trace events are timed on the input
    timeline and drift from real arrival time when a round overruns). The
    written PCM itself is at true arrival time, so re-deriving segments from
    it is model-agnostic.

    Method: windowed RMS (PCM normalized to [-1, 1]) thresholded per
    `window_s` window, then:
      1. `merge_gap_s` bridges short silences between speech windows, since
         chunked APIs often deliver one utterance in bursts with brief empty
         rounds in between.
      2. `min_segment_s` drops blips too short to be speech.

    `threshold_rms=0.01` is about 20 dB below typical speech level (~0.1)
    and well above digital silence.
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
    (stop / hold_floor / interrupt_user / respond_and_recover), where a live
    trigger controller decides mid-stream when to append the next clip
    (e.g. cue.wav).

    Sender-loop usage:

        i = 0
        while not source.is_done:
            chunk = source.pull(CHUNK_BYTES)
            await ws.send(...chunk...)
            await sleep_until_deadline(loop, send_t0 + (i + 1) * (CHUNK_MS / 1000))
            i += 1

    Semantics:
      - `append(pcm_bytes)` queues more content (lead_in.wav upfront, then
        cue.wav when the controller triggers).
      - `pull(n)` always returns exactly n bytes immediately, padding with
        digital silence when the buffer is short, so waiting for a trigger
        sounds like silence rather than a stalled stream.
      - `finish()` marks that nothing more will be appended; `is_done` is
        True once finished and the buffer is empty. Callers that want a
        post-clip silence tail should stop pulling on their own timer.
    """

    def __init__(self, initial_pcm: bytes = b""):
        self._buffer = bytearray(initial_pcm)
        self._finished = False

    def append(self, pcm_bytes: bytes) -> None:
        self._buffer.extend(pcm_bytes)

    def truncate_pending(self) -> int:
        """
        Drop all not-yet-pulled content and return the number of bytes
        dropped. Used by `interrupt_user` (see interrupt_controller.py): when
        the model barges in, the simulated user stops talking immediately
        instead of finishing the queued clip. Does not call `finish()`;
        callers append the next clip (e.g. a branch reply) and finish as usual.
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

    Tokens are joined with no separator: the word/subword streams of Moshi,
    FLM-Audio, BayLing, MiniCPM-o, and Raon already carry a leading space at
    word boundaries. A client whose tokens do not self-space should override
    `compute_model_speech_transcripts()` instead.

    Must stay identical to `stop_common.default_model_speech_transcripts()`
    in simulator_replay: `run_metrics` scores both trees off this field.
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
    recorder wiring, event logging, timing, and result shape are shared so
    results are comparable across models.
    """

    MODEL_SLUG: str = "unset"    # directory-safe key, must match model_registry.py
    OUTPUT_SR: int = 24_000      # sample rate of audio returned by the model
    INPUT_SR: int = 24_000       # sample rate the API expects for input audio
    # See SynchronizedRecorder.contiguous_while_speaking.
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
        Run one sample: wire the recorder and event logger, call
        `_connect_and_stream`, and return the "inference" dict that
        run_inference.py merges with instance metadata.

        t=0 for the recorder and events.jsonl is when input streaming begins,
        not when the connection opens (handshake events get negative
        timestamps), so all timestamps align with metadata.json's
        input.wav-relative times (pause_start, bc_start, ...).
        """
        setup_t0 = time.monotonic()
        event_logger = EventLogger(events_path, setup_t0)
        recorder = SynchronizedRecorder(self.OUTPUT_SR, input_duration, output_wav)
        recorder.contiguous_while_speaking = bool(self.CONTIGUOUS_WHILE_SPEAKING)

        stream_start: dict = {"t": None}
        recorder_task_holder: dict = {"task": None}

        async def start_streaming() -> float:
            """
            Call from `_connect_and_stream` immediately before sending the
            first input chunk (idempotent, safe on reconnect). Sets t=0 for
            the recorder and event log and returns that monotonic value, so
            the sender can anchor its deadline pacing to the same clock read.
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
            # External cancellation, typically run_inference.py's per-sample
            # wait_for timeout. Re-raise so wait_for sees it; the finally
            # block skips real-time padding so teardown is immediate.
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
            # received == accepted + dropped; dropped counts late deltas
            # from already-cancelled responses.
            "audio_bytes_received":  recorder.bytes_received,
            "audio_bytes_accepted":  recorder.bytes_accepted,
            "audio_bytes_dropped":   recorder.bytes_dropped,
            "errors":                errors,
            "inference_wallclock_s": round(wallclock, 4),
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }

    def _model_speech_transcripts(self, segments: list) -> list[str]:
        """
        The model's native text for each speech segment, used instead of
        Whisper for the slugs in `run_metrics.LIVE_TRANSCRIPT_MODELS` (BayLing
        and Raon, whose output.wav is mostly silence padding and defeats ASR).

        A client opts in by accumulating `self.text_tokens` as (t, text) on the
        recorder's clock, or by overriding
        `compute_model_speech_transcripts(segments)`. Otherwise returns "" per
        segment; the length always matches `segments`, because
        `run_metrics.segment_texts()` checks `len(native) == len(segs)`.
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

        The base class pads to input_duration if this returns early.
        """
        raise NotImplementedError
