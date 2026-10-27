"""
Temporal Ordering: build a 4-way multiple-choice task from any directory of audio.

Each item splits a clip into N equal chronological segments (default 3), plays them in a
shuffled order, and asks for the original order:

    Segment 1: <audio>
    Segment 2: <audio>
    Segment 3: <audio>
    (A) 2, 3, 1   (B) ...   (C) ...   (D) ...

Each option lists the playback indices in chronological order. Distractors are other
permutations, so sources are processed one at a time and memory stays constant.

Permutation semantics:
  shuffle_permutation[p] (1-based) = the chronological segment played at position p.
  answer_order = the inverse: for each chronological position, the playback index that
                 holds it. e.g. shuffle_permutation [3, 1, 2] -> answer_order "2, 3, 1".

Output: <out-dir>/audio/*.wav and <out-dir>/<out-jsonl-name>. Each row has an ordered
`audios` list (absolute paths, in playback order) matching the <audio> placeholders in
`question`, and the correct option in `answer_letter` / `label`.

Usage:
  python make_ordering.py --audio-dir <wav_dir> --out-dir <out>/ordering
"""

import argparse
import json
import math
import os
import random
import sys
from itertools import permutations

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from common import add_source_args, list_sources, source_id

TARGET_SR    = 16000
NUM_SEGMENTS = 3
NUM_CHOICES  = 4          # 1 correct order + (NUM_CHOICES - 1) distractor orders
TASK_NAME    = "ordering"
SEED         = 1234

# Duration window in seconds (<= 0 disables a bound); same as the inpainting task.
MIN_DURATION_S = 3.0
MAX_DURATION_S = 12.0


def build_question_template(n: int) -> str:
    """Prompt with one <audio> placeholder per segment (in playback order) and a
    {choices} block."""
    seg_lines = "".join(f"Segment {i + 1}: <audio>\n" for i in range(n))
    return (
        f"You are given {n} audio segments cut from a single continuous "
        "conversation.\n"
        "They are played in a randomly shuffled order and numbered by that "
        "playback order:\n"
        "\n"
        + seg_lines +
        "\n"
        f"Your task is to recover the ORIGINAL chronological order of the {n} "
        "segments. Each option below lists the playback segment indices in "
        "chronological order.\n"
        "\n"
        "Reason step by step:\n"
        "Step 1 — Transcribe each segment in full, labeled by its playback number.\n"
        "Step 2 — Find cues that link the segments: question-answer pairs, "
        "discourse markers, pronoun references, topic and turn continuity, "
        "sentences that begin in one segment and finish in another, speaker "
        "turn-taking, and prosody.\n"
        "Step 3 — Decide which playback segment came first, second, and so on, "
        "then match your conclusion to one of the options.\n"
        "\n"
        "Which option gives the correct chronological order?\n"
        "{choices}"
    )


def split_into_segments(wave: np.ndarray, n: int) -> list:
    """Split into n equal-length chronological chunks (the last takes the remainder)."""
    total = wave.shape[0]
    seg_len = total // n
    segs = []
    for i in range(n):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n - 1 else total
        segs.append(wave[start:end])
    return segs


def make_permutation(rng: random.Random, n: int, allow_identity: bool) -> list:
    """0-indexed permutation: playback position p -> chronological segment."""
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if allow_identity or any(perm[i] != i for i in range(n)):
            return list(perm)


def answer_from_perm(perm0: list) -> list:
    """Inverse permutation (1-indexed): for each chronological position, the playback
    index that holds it."""
    inv = [0] * len(perm0)
    for p, c in enumerate(perm0):
        inv[c] = p + 1
    return inv


def build_choices(answer_order: str, n: int, n_choices: int, rng_choice):
    """Correct order + (n_choices - 1) distractor orders, shuffled.
    Returns (choices_block, choice_orders, answer_letter)."""
    all_orders = [", ".join(map(str, p)) for p in permutations(range(1, n + 1))]
    distractors = [o for o in all_orders if o != answer_order]
    rng_choice.shuffle(distractors)
    chosen = distractors[: n_choices - 1] + [answer_order]
    rng_choice.shuffle(chosen)
    letters = [chr(ord('A') + i) for i in range(len(chosen))]
    choice_orders = dict(zip(letters, chosen))
    answer_letter = letters[chosen.index(answer_order)]
    block = "\n".join(f"({l}) {o}" for l, o in choice_orders.items())
    return block, choice_orders, answer_letter


def run(args):
    n_seg = args.num_segments
    n_choices = args.num_choices
    if math.factorial(n_seg) < n_choices:
        sys.exit(f"[error] num-segments={n_seg} yields only {math.factorial(n_seg)} "
                 f"distinct orders < num-choices={n_choices}")

    # Absolute paths, so the trainer finds the audio regardless of its working directory.
    out_audio_dir = os.path.abspath(os.path.join(args.out_dir, "audio"))
    os.makedirs(out_audio_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, args.out_jsonl_name)

    src_files = list_sources(args)
    if args.limit and args.limit > 0:
        src_files = src_files[: args.limit]

    rng = random.Random(args.seed)
    # Separate RNG for the options, so the segment shuffle (and the written audio) does
    # not depend on how the options are sampled.
    rng_choice = random.Random(args.seed + 12345)
    question_tmpl = build_question_template(n_seg)

    n_written = n_failed = n_filtered = 0
    letter_dist = {chr(ord('A') + i): 0 for i in range(n_choices)}

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for src_path in tqdm(src_files, desc="Building ordering items"):
            stem = source_id(src_path, args.audio_dir)

            # Duration check before decoding.
            try:
                dur = librosa.get_duration(path=src_path)
            except Exception as e:
                print(f"[dur-fail] {src_path}: {e}", file=sys.stderr)
                n_failed += 1
                continue
            if args.min_duration > 0 and dur < args.min_duration:
                n_filtered += 1
                continue
            if args.max_duration > 0 and dur > args.max_duration:
                n_filtered += 1
                continue

            try:
                wave, sr = librosa.load(src_path, sr=TARGET_SR, mono=True)
            except Exception as e:
                print(f"[load-fail] {src_path}: {e}", file=sys.stderr)
                n_failed += 1
                continue

            chrono_segs = split_into_segments(wave, n_seg)
            if any(s.shape[0] == 0 for s in chrono_segs):
                n_failed += 1
                continue

            perm0 = make_permutation(rng, n_seg, args.allow_identity)

            seg_paths = []                                  # in playback order
            for p, c in enumerate(perm0):
                out_path = os.path.join(out_audio_dir, f"{stem}__seg{p + 1}.wav")
                sf.write(out_path, chrono_segs[c], sr, subtype="PCM_16")
                seg_paths.append(out_path)

            answer_order = ", ".join(str(x) for x in answer_from_perm(perm0))
            choices_block, choice_orders, answer_letter = build_choices(
                answer_order, n_seg, n_choices, rng_choice)
            letter_dist[answer_letter] += 1

            row = {
                "id": f"{stem}__ordering",
                "source_id": stem,
                "ssl_task": TASK_NAME,
                "question": question_tmpl.format(choices=choices_block),
                "audios": seg_paths,
                "answer_letter": answer_letter,
                "label": answer_letter,
                "answer_order": answer_order,                   # e.g. "2, 3, 1"
                "choice_orders": choice_orders,                 # {"A": "1, 2, 3", ...}
                "num_choices": n_choices,
                "num_segments": n_seg,
                "shuffle_permutation": [c + 1 for c in perm0],  # playback -> chronological
                "segment_duration_s": round((wave.shape[0] // n_seg) / sr, 4),
                "original_duration_s": round(wave.shape[0] / sr, 4),
                "sample_rate": sr,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[done] wrote={n_written}  filtered(out-of-range)={n_filtered}  "
          f"failed/skipped={n_failed}")
    print(f"[done] answer-letter distribution: {letter_dist}")
    print(f"[done] jsonl -> {out_jsonl}")
    print(f"[done] audio -> {out_audio_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Build the Temporal Ordering task (N-way MCQ) from a directory of audio.")
    add_source_args(ap)
    ap.add_argument("--out-dir",        required=True)
    ap.add_argument("--out-jsonl-name", default="train.jsonl")
    ap.add_argument("--seed",           type=int,   default=SEED)
    ap.add_argument("--num-segments",   type=int,   default=NUM_SEGMENTS)
    ap.add_argument("--num-choices",    type=int,   default=NUM_CHOICES)
    ap.add_argument("--min-duration",   type=float, default=MIN_DURATION_S)
    ap.add_argument("--max-duration",   type=float, default=MAX_DURATION_S)
    ap.add_argument("--allow-identity", action="store_true",
                    help="allow the unshuffled order (a trivial item); off by default")
    ap.add_argument("--limit",          type=int,   default=None,
                    help="use only the first N source files (for testing)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
