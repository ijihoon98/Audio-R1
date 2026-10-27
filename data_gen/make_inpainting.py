"""
Region Inpainting: build a 4-way multiple-choice task from any directory of audio.

Each item presents a clip whose middle third is replaced with silence, plus four candidate
segments. The model picks the candidate that fills the gap:

    You are given an audio clip with one missing part.
    <clip = before + [silence] + after>
    (A) <audio>  (B) <audio>  (C) <audio>  (D) <audio>

Construction (per source):
  1. Load 16 kHz mono and split into three equal parts: before / missing / after.
  2. Clip = before + silence(len(missing)) + after; the correct candidate = missing.
  3. Three distractors = centre crops of the same length from three other sources, so
     every candidate has the same duration.
  4. Shuffle the four candidates and record the correct letter.

Output: <out-dir>/audio/*.wav and <out-dir>/<out-jsonl-name>. Each row has an ordered
`audios` list (absolute paths: clip, A, B, C, D) matching the five <audio> placeholders in
`question`, and the correct option in `answer_letter` / `label`.

Note: all usable sources are held in memory. For a very large corpus, subsample it with
--max-sources.

Usage:
  python make_inpainting.py --audio-dir <wav_dir> --out-dir <out>/inpainting
"""

import argparse
import json
import os
import random
import sys

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from common import add_source_args, list_sources, source_id

TARGET_SR    = 16000
NUM_SEGMENTS = 3          # before / missing / after
NUM_CHOICES  = 4          # 1 correct + 3 distractors
TASK_NAME    = "inpainting"
SEED         = 1234

# Duration window in seconds (<= 0 disables a bound).
MIN_DURATION_S = 3.0
MAX_DURATION_S = 12.0

# Five <audio> placeholders: the clip, then the four candidates.
QUESTION_TEMPLATE = (
    "You are given an audio clip with one missing part.\n"
    "<audio>\n"
    "\n"
    "Task:\n"
    "Determine the candidate audio segment that best fits the missing part of "
    "the original audio.\n"
    "Select the correct answer from the following 4 choices:\n"
    "(A) <audio>\n"
    "(B) <audio>\n"
    "(C) <audio>\n"
    "(D) <audio>\n"
    "\n"
    "The missing part should be inferred from the surrounding audio context, "
    "such as speech content, speaker identity, acoustic continuity, background "
    "sound, and temporal flow."
)


def split3(wave: np.ndarray):
    """Split into 3 equal chunks (the last takes the remainder): before, middle, after."""
    n = wave.shape[0]
    s = n // NUM_SEGMENTS
    return wave[:s], wave[s:2 * s], wave[2 * s:]


def middle_window(wave: np.ndarray, length: int):
    """Centre crop of `length` samples; None if too short."""
    n = wave.shape[0]
    if n < length:
        return None
    start = (n - length) // 2
    return wave[start:start + length]


def run(args):
    # Absolute paths, so the trainer finds the audio regardless of its working directory.
    out_audio_dir = os.path.abspath(os.path.join(args.out_dir, "audio"))
    os.makedirs(out_audio_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, args.out_jsonl_name)

    src_files = list_sources(args)
    if args.limit and args.limit > 0:
        src_files = src_files[: args.limit]

    # Load every usable source once: the full waveform is needed both for the split and
    # for cropping distractors.
    waves = {}
    for path in tqdm(src_files, desc="Loading sources"):
        try:
            wave, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"[load-fail] {path}: {e}", file=sys.stderr)
            continue
        dur_s = wave.shape[0] / TARGET_SR
        if args.min_duration > 0 and dur_s < args.min_duration:
            continue
        if args.max_duration > 0 and dur_s > args.max_duration:
            continue
        waves[path] = wave

    valid_paths = sorted(waves.keys())
    print(f"[info] usable sources after duration filter: {len(valid_paths)}")
    if len(valid_paths) < NUM_CHOICES:
        sys.exit("[error] too few usable sources; check --audio-dir and --min/--max-duration")

    rng = random.Random(args.seed)
    n_written = n_failed = 0
    letter_dist = {chr(ord('A') + i): 0 for i in range(NUM_CHOICES)}

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for src_path in tqdm(valid_paths, desc="Building inpainting items"):
            stem = source_id(src_path, args.audio_dir)
            try:
                wave = waves[src_path]
                before, missing, after = split3(wave)
                missing_len = missing.shape[0]
                if missing_len == 0 or before.shape[0] == 0 or after.shape[0] == 0:
                    n_failed += 1
                    continue

                # The clip with its middle replaced by silence of the same length.
                hole = np.zeros(missing_len, dtype=wave.dtype)
                clip = np.concatenate([before, hole, after], axis=0)

                # Distractors: same-length centre crops of other sources.
                cand_pool = [p for p in valid_paths if p != src_path]
                rng.shuffle(cand_pool)
                distractors = []
                distractor_ids = []
                for other in cand_pool:
                    seg = middle_window(waves[other], missing_len)
                    if seg is None:
                        continue
                    distractors.append(seg)
                    distractor_ids.append(source_id(other, args.audio_dir))
                    if len(distractors) == NUM_CHOICES - 1:
                        break
                if len(distractors) < NUM_CHOICES - 1:
                    n_failed += 1
                    continue

                candidates = [("__correct__", missing)] + list(zip(distractor_ids, distractors))
                rng.shuffle(candidates)
                correct_idx = next(i for i, (sid, _) in enumerate(candidates)
                                   if sid == "__correct__")
                answer_letter = chr(ord('A') + correct_idx)
                letter_dist[answer_letter] += 1

                clip_path = os.path.join(out_audio_dir, f"{stem}__clip.wav")
                sf.write(clip_path, clip, TARGET_SR, subtype="PCM_16")
                choice_paths = []
                choice_source_ids = []
                for i, (sid, seg) in enumerate(candidates):
                    letter = chr(ord('A') + i)
                    cpath = os.path.join(out_audio_dir, f"{stem}__cand{letter}.wav")
                    sf.write(cpath, seg, TARGET_SR, subtype="PCM_16")
                    choice_paths.append(cpath)
                    choice_source_ids.append(stem if sid == "__correct__" else sid)

                row = {
                    "id": f"{stem}__inpainting",
                    "source_id": stem,
                    "ssl_task": TASK_NAME,
                    "question": QUESTION_TEMPLATE,
                    "audios": [clip_path] + choice_paths,    # clip, A, B, C, D
                    "answer_letter": answer_letter,
                    "label": answer_letter,
                    "choice_source_ids": choice_source_ids,
                    "num_choices": NUM_CHOICES,
                    "missing_duration_s": round(missing_len / TARGET_SR, 4),
                    "original_duration_s": round(wave.shape[0] / TARGET_SR, 4),
                    "sample_rate": TARGET_SR,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            except Exception as e:
                print(f"[process-fail] {stem}: {e}", file=sys.stderr)
                n_failed += 1
                continue

    print(f"[done] wrote={n_written}  failed/skipped={n_failed}")
    print(f"[done] answer-letter distribution: {letter_dist}")
    print(f"[done] jsonl -> {out_jsonl}")
    print(f"[done] audio -> {out_audio_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Build the Region Inpainting task (4-way MCQ) from a directory of audio.")
    add_source_args(ap)
    ap.add_argument("--out-dir",        required=True)
    ap.add_argument("--out-jsonl-name", default="train.jsonl")
    ap.add_argument("--seed",           type=int,   default=SEED)
    ap.add_argument("--min-duration",   type=float, default=MIN_DURATION_S)
    ap.add_argument("--max-duration",   type=float, default=MAX_DURATION_S)
    ap.add_argument("--limit",          type=int,   default=None,
                    help="use only the first N source files (for testing)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
