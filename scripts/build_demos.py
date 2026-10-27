"""Build the Demo section data for the project page.

Picks a few SSL-task test items with clean, speech-rich audio, converts their audio to MP3 under
static/audio/demo/, and writes static/js/demos.js (window.DEMOS = [...]).

Usage: python scripts/build_demos.py
"""
import json
import os
import subprocess

BENCH = "/lustre09/project/6080535/jihoojo3/SSL-R1/benchmarks"
SITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_OUT = os.path.join(SITE, "static", "audio", "demo")
JS_OUT = os.path.join(SITE, "static", "js", "demos.js")

TASKS = {
    "identity": ("MMAU_temporal_ssl_identity_v2", "mmau_identity_v2_ssl.jsonl"),
    "inpainting": ("MMAU_temporal_ssl_inpainting", "mmau_inpainting_ssl.jsonl"),
    "ordering": ("MMAU_temporal_ssl_3seg", "mmau_temporal_ssl_3seg.jsonl"),
}

# (task, source-id prefix). Chosen by simple signal checks on every clip (speech activity >= 60%,
# level >= -32 dBFS, no clipping, no pause > 0.8 s), all from MMAU's synthetic subset; 8 per task,
# at most 3 sharing the same answer. Ordered so neighbouring examples never repeat the correct answer
# (letter or option text), and identity examples never show the same option text in the same slot.
PICKS = [
    ("identity", "9b7323cf"),
    ("identity", "86cd240d"),
    ("identity", "98f8b556"),
    ("identity", "a6f1c137"),
    ("identity", "ed934345"),
    ("identity", "aba65a16"),
    ("identity", "18fd5726"),
    ("identity", "e4db0e63"),
    ("inpainting", "7235d35a"),
    ("inpainting", "38dbb2bf"),
    ("inpainting", "6d3e431a"),
    ("inpainting", "12ea6970"),
    ("inpainting", "f4d842a3"),
    ("inpainting", "9685984a"),
    ("inpainting", "976cd4b0"),
    ("inpainting", "daa1f53f"),
    ("ordering", "e4db0e63"),
    ("ordering", "24d64f05"),
    ("ordering", "38dbb2bf"),
    ("ordering", "ea8a2fc9"),
    ("ordering", "cde380fe"),
    ("ordering", "06483ca8"),
    ("ordering", "aac87e2d"),
    ("ordering", "b4180fa8"),
]

IDENTITY_TEXT = {
    "only1": "Only Candidate 1 is the same recording",
    "only2": "Only Candidate 2 is the same recording",
    "both_same": "Both candidates are the same recording",
    "both_diff": "Neither candidate is the same recording",
}


def to_mp3(src, name):
    dst = os.path.join(AUDIO_OUT, name)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-ac", "1", "-ar", "16000", "-b:a", "48k", dst], check=True)
    return f"static/audio/demo/{name}"


def load_by_prefix(path, jsonl=False):
    rows = [json.loads(l) for l in open(path)] if jsonl else json.load(open(path))
    return {r["id"][:8]: r for r in rows}


def main():
    os.makedirs(AUDIO_OUT, exist_ok=True)
    demos = []
    for task, pid in PICKS:
        folder, qfile = TASKS[task]
        q = load_by_prefix(os.path.join(BENCH, folder, "question", qfile), jsonl=True)[pid]
        adir = os.path.join(BENCH, folder, "audio")
        tag = f"{task}_{pid}"

        span = None  # (start, width) of the masked region as fractions of the clip; inpainting only
        extra = {}
        if task == "identity":
            inputs = [("Anchor", q["anchor_file"])] + [
                (f"Candidate {i + 1}", f) for i, f in enumerate(q["candidate_files"])]
            options = {k: IDENTITY_TEXT[v] for k, v in q["choice_states"].items()}
            question = "Which candidates are the same recording as the anchor?"
            candidates = []
            # whether each candidate is an exact copy of the anchor (used for colouring/badges)
            extra["same"] = [bool(q["cand1_same"]), bool(q["cand2_same"])]
        elif task == "inpainting":
            inputs = [("Masked clip", q["clip_file"])]
            candidates = [(chr(ord("A") + i), f) for i, f in enumerate(q["choice_files"])]
            options = {k: f"Candidate {k}" for k, _ in candidates}
            # the middle segment of the clip is replaced with silence (Sec. 3.2 of the paper)
            width = q["missing_duration_s"] / q["original_duration_s"]
            span = [round((1 - width) / 2, 4), round(width, 4)]
            question = "Which candidate fills the silent gap in the clip?"
        else:
            inputs = [(f"Segment {i + 1}", f) for i, f in enumerate(q["segment_files"])]
            options = {k: v.replace(", ", " → ") for k, v in q["choice_orders"].items()}
            question = "What is the original chronological order of the shuffled segments?"
            candidates = []
            # playback indices (1-based) listed in chronological order
            extra["order"] = [int(i) for i in q["answer_order"].split(",")]

        demos.append({
            "id": tag,
            "task": task,
            "question": question,
            "inputs": [{"label": lab, "src": to_mp3(os.path.join(adir, f), f"{tag}_{i}.mp3"),
                        **({"gap": span} if span else {})}
                       for i, (lab, f) in enumerate(inputs)],
            # candidates are drawn to scale, under the gap they would fill
            "candidates": [{"label": lab, "src": to_mp3(os.path.join(adir, f), f"{tag}_cand{lab}.mp3"),
                            "span": span}
                           for lab, f in candidates],
            "options": options,
            "answer": q["answer_gt"],
            **extra,
        })

    with open(JS_OUT, "w") as f:
        f.write("// Generated by scripts/build_demos.py -- do not edit by hand.\n")
        f.write("window.DEMOS = " + json.dumps(demos, indent=2, ensure_ascii=False) + ";\n")
    print(f"wrote {len(demos)} demos -> {JS_OUT}")


if __name__ == "__main__":
    main()
