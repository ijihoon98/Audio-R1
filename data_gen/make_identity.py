"""
Identity Detection: build a 4-way multiple-choice task from any directory of audio.

Each item presents an anchor clip and two candidate clips, and asks which candidates are
the same recording as the anchor. The four mutually exclusive outcomes form the options:

    Anchor:      <audio>
    Candidate 1: <audio>
    Candidate 2: <audio>

    (A..D, shuffled per item)
    - Both candidates are the same recording as the Anchor.
    - Only Candidate 1 is the same recording as the Anchor.
    - Only Candidate 2 is the same recording as the Anchor.
    - Neither candidate is the same recording as the Anchor.

Construction (anchor = clip of source i):
  both_same : C1 = copy of anchor, C2 = copy of anchor
  only1     : C1 = copy of anchor, C2 = clip of source j != i
  only2     : C1 = clip of source j != i, C2 = copy of anchor
  both_diff : C1 = clip of source j, C2 = clip of source k (j, k != i, j != k)

Shortcut removal: every clip is cropped to the same length, the four outcomes are balanced
(~25% each), and the option texts are shuffled across A..D for every item.

Output: <out-dir>/audio/*.wav and <out-dir>/<out-jsonl-name>. Each row has an ordered
`audios` list (absolute paths) matching the <audio> placeholders in `question`, and the
correct option in `answer_letter` / `label`.

Usage:
  python make_identity.py --audio-dir <wav_dir> --out-dir <out>/identity
"""

import argparse
import json
import os
import random
import sys

import librosa
import soundfile as sf
from tqdm import tqdm

from common import add_source_args, list_sources, source_id

TARGET_SR       = 16000
CLIP_DURATION_S = 5.0     # fixed clip length -> no duration shortcut
TASK_NAME       = "identity"
SEED            = 1234

# The four ground-truth states and their (cand1_same, cand2_same) flags.
STATES = ["both_same", "only1", "only2", "both_diff"]
STATE_FLAGS = {
    "both_same": (True, True),
    "only1":     (True, False),
    "only2":     (False, True),
    "both_diff": (False, False),
}

# Three <audio> placeholders (anchor, candidate 1, candidate 2); {choices} is filled per row.
QUESTION_TEMPLATE = (
    "You are given an anchor audio clip and two candidate audio clips:\n"
    "Anchor: <audio>\n"
    "Candidate 1: <audio>\n"
    "Candidate 2: <audio>\n"
    "For each candidate, decide whether it is the same recording as the Anchor "
    "(an exact copy) or comes from a different source recording.\n"
    "Which option is correct?\n"
    "{choices}"
)

STATE_DESC = {
    "both_same": "Both Candidate 1 and Candidate 2 are the same recording as "
                 "the Anchor.",
    "only1":     "Only Candidate 1 is the same recording as the Anchor "
                 "(Candidate 2 is from a different recording).",
    "only2":     "Only Candidate 2 is the same recording as the Anchor "
                 "(Candidate 1 is from a different recording).",
    "both_diff": "Neither candidate is the same recording as the Anchor.",
}


def build_choices(answer_state: str, rng_choice):
    """Shuffle the four state descriptions across A..D.
    Returns (choices_block, choice_states, answer_letter)."""
    order = list(STATES)
    rng_choice.shuffle(order)
    letters = ["A", "B", "C", "D"]
    choice_states = dict(zip(letters, order))          # letter -> state
    answer_letter = letters[order.index(answer_state)]
    block = "\n".join(f"({l}) {STATE_DESC[st]}" for l, st in choice_states.items())
    return block, choice_states, answer_letter


def load_clip(path: str, clip_n: int):
    """Load 16 kHz mono and keep the first clip_n samples; None if too short."""
    wave, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if wave.shape[0] < clip_n:
        return None
    return wave[:clip_n]


def balanced_states(n: int, rng) -> list:
    """Length-n list of states, as close to 25% each as possible, shuffled."""
    base = n // len(STATES)
    states = STATES * base
    states += rng.sample(STATES, n - len(states))  # spread the remainder
    rng.shuffle(states)
    return states


def run(args):
    # Absolute paths, so the trainer finds the audio regardless of its working directory.
    out_audio_dir = os.path.abspath(os.path.join(args.out_dir, "audio"))
    os.makedirs(out_audio_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, args.out_jsonl_name)

    clip_n = int(round(args.clip_duration * TARGET_SR))

    src_files = list_sources(args)

    # Keep only sources long enough for a full clip. Only paths are stored, so memory
    # stays constant on large directories.
    pool = []
    for path in tqdm(src_files, desc="Scanning durations"):
        try:
            if librosa.get_duration(path=path) >= args.clip_duration:
                pool.append((source_id(path, args.audio_dir), path))
        except Exception as e:
            print(f"[dur-fail] {path}: {e}", file=sys.stderr)
    if args.limit and args.limit > 0:
        pool = pool[: args.limit]
    print(f"[info] qualifying sources (>= {args.clip_duration:g}s): {len(pool)}")
    if len(pool) < 3:
        sys.exit("[error] need >= 3 qualifying sources for the 'both_diff' case")

    rng = random.Random(args.seed)
    # Separate RNG for the A..D order, so the pairing (and the written audio) does not
    # depend on how the options are shuffled.
    rng_choice = random.Random(args.seed + 12345)
    state_plan = balanced_states(len(pool), rng)

    def pick_other(exclude: set) -> int:
        """Random pool index not in `exclude`."""
        j = rng.randrange(len(pool))
        while j in exclude:
            j = rng.randrange(len(pool))
        return j

    n_written = n_failed = 0
    state_count = {s: 0 for s in STATES}

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for i, (stem, path) in enumerate(tqdm(pool, desc="Building identity items")):
            state = state_plan[i]
            cand1_same, cand2_same = STATE_FLAGS[state]
            try:
                anchor = load_clip(path, clip_n)
                if anchor is None:
                    n_failed += 1
                    continue

                # A candidate is a copy of the anchor when "same", otherwise a clip
                # from a different source.
                used = {i}
                cand1_src = cand2_src = stem
                if cand1_same:
                    cand1 = anchor
                else:
                    j = pick_other(used)
                    cand1 = load_clip(pool[j][1], clip_n)
                    if cand1 is None:
                        n_failed += 1
                        continue
                    used.add(j)
                    cand1_src = pool[j][0]

                if cand2_same:
                    cand2 = anchor
                else:
                    k = pick_other(used)
                    cand2 = load_clip(pool[k][1], clip_n)
                    if cand2 is None:
                        n_failed += 1
                        continue
                    used.add(k)
                    cand2_src = pool[k][0]

                pa = os.path.join(out_audio_dir, f"{stem}__anchor.wav")
                p1 = os.path.join(out_audio_dir, f"{stem}__cand1.wav")
                p2 = os.path.join(out_audio_dir, f"{stem}__cand2.wav")
                sf.write(pa, anchor, TARGET_SR, subtype="PCM_16")
                sf.write(p1, cand1, TARGET_SR, subtype="PCM_16")
                sf.write(p2, cand2, TARGET_SR, subtype="PCM_16")

                choices_block, choice_states, answer_letter = build_choices(
                    state, rng_choice)

                row = {
                    "id": f"{stem}__identity",
                    "source_id": stem,
                    "ssl_task": TASK_NAME,
                    "question": QUESTION_TEMPLATE.format(choices=choices_block),
                    "audios": [pa, p1, p2],                # anchor, candidate 1, candidate 2
                    "answer_letter": answer_letter,
                    "label": answer_letter,
                    "answer_state": state,                 # both_same | only1 | only2 | both_diff
                    "cand1_same": cand1_same,
                    "cand2_same": cand2_same,
                    "choice_states": choice_states,        # {"A": "only2", ...}
                    "num_choices": 4,
                    "clip_duration_s": args.clip_duration,
                    "cand1_source_id": cand1_src,
                    "cand2_source_id": cand2_src,
                    "sample_rate": TARGET_SR,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                state_count[state] += 1
            except Exception as e:
                print(f"[process-fail] {stem}: {e}", file=sys.stderr)
                n_failed += 1
                continue

    print(f"[done] wrote={n_written}  failed/skipped={n_failed}")
    print(f"[done] state distribution: {state_count}")
    print(f"[done] jsonl -> {out_jsonl}")
    print(f"[done] audio -> {out_audio_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Build the Identity Detection task (4-way MCQ) from a directory of audio.")
    add_source_args(ap)
    ap.add_argument("--out-dir",        required=True)
    ap.add_argument("--out-jsonl-name", default="train.jsonl")
    ap.add_argument("--seed",           type=int,   default=SEED)
    ap.add_argument("--clip-duration",  type=float, default=CLIP_DURATION_S,
                    help="fixed clip length in seconds (shorter sources are skipped)")
    ap.add_argument("--limit",          type=int,   default=None,
                    help="use only the first N qualifying sources (for testing)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
