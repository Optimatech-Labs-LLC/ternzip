"""Offline command line entry point used by the relocatable Mac preview."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec import pack
from chat import run
from layouts import FORMATS, restore_layout


def main():
    parser = argparse.ArgumentParser(prog="ternzip-27b", description="Lossless pack of Prism's 27B language GGUF")
    subs = parser.add_subparsers(dest="action", required=True)
    unpack = subs.add_parser("restore", help="Recreate an official ternary GGUF and verify Prism's SHA-256")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("restored", type=Path)
    unpack.add_argument("--receipt", type=Path)
    unpack.add_argument("--format", dest="fmt", choices=tuple(FORMATS), default="PTQ1_0")
    build = subs.add_parser("pack", help="Build the archive from the official PTQ1_0 file")
    build.add_argument("source", type=Path)
    build.add_argument("archive", type=Path)
    build.add_argument("receipt", type=Path)
    chat = subs.add_parser("chat", help="Run one prompt from the packed archive on Apple Metal")
    chat.add_argument("path", type=Path)
    chat.add_argument("prompt")
    chat.add_argument("--limit", type=int, default=128)
    chat.add_argument("--receipt", type=Path)
    kwargs = vars(parser.parse_args())
    action = kwargs.pop("action")
    if action == "restore":
        restore_layout(**kwargs)
    elif action == "pack":
        pack(**kwargs)
    else:
        if not 1 <= kwargs["limit"] <= 768:
            parser.error("--limit must be between 1 and 768")
        run(fmt="PQ2_0", rounds=1, verbose=False, **kwargs)


if __name__ == "__main__":
    main()
