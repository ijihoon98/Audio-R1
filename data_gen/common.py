"""Source discovery shared by the task generators."""

import os
import random
import sys
from glob import glob

EXTENSIONS = ["wav", "mp3", "flac"]


def add_source_args(ap):
    """--audio-dir / --extensions / --max-sources, shared by every generator."""
    ap.add_argument("--audio-dir", required=True,
                    help="directory of source audio, searched recursively (any layout)")
    ap.add_argument("--extensions", nargs="+", default=EXTENSIONS)
    ap.add_argument("--max-sources", type=int, default=None,
                    help="randomly sample at most N source files (seeded by --seed)")


def list_sources(args):
    """Sorted audio files under --audio-dir (recursive), optionally subsampled."""
    files = sorted({f for ext in args.extensions
                    for f in glob(os.path.join(args.audio_dir, "**", f"*.{ext}"),
                                  recursive=True)})
    if not files:
        sys.exit(f"[error] no audio under --audio-dir: {args.audio_dir}")
    if args.max_sources and args.max_sources < len(files):
        files = sorted(random.Random(args.seed).sample(files, args.max_sources))
    print(f"[info] source files: {len(files)}")
    return files


def source_id(path, audio_dir):
    """Unique id from the path relative to --audio-dir, e.g. spk/chap/utt.wav -> spk_chap_utt."""
    rel = os.path.splitext(os.path.relpath(path, audio_dir))[0]
    return rel.replace(os.sep, "_")
