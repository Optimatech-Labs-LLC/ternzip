"""Set up a local runtime and dispatch a model file operation."""
import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def setup_lock(cache):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "setup.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def dependencies_ready():
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        try:
            if importlib.metadata.version(name) != version:
                return False
        except importlib.metadata.PackageNotFoundError:
            return False
    return True


def main():
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("This release uses CPython 3.14 with compression.zstd.")
    from archive_io import ARCHIVE_NAME
    parser = argparse.ArgumentParser(prog="ternzip", description="Lossless ternary model archive tools.")
    actions = parser.add_subparsers(dest="command", required=True)
    restore_cmd = actions.add_parser("restore", help="Write and verify the exact official GGUF")
    restore_cmd.add_argument("archive", type=Path)
    restore_cmd.add_argument("output", type=Path, nargs="?", help="Defaults to Prism's original filename")
    verify = actions.add_parser("verify", help="Restore into a checksum without writing a GGUF")
    verify.add_argument("archive", type=Path)
    pack = actions.add_parser("pack", help="Rebuild the 8B archive using the released 8B encoder profile")
    pack.add_argument("gguf", type=Path)
    pack.add_argument("output", type=Path, nargs="?", default=Path(ARCHIVE_NAME))
    chat_cmd = actions.add_parser("chat", help="Generate a reply on an Apple Silicon Mac")
    chat_cmd.add_argument("archive", type=Path)
    chat_cmd.add_argument("prompt")
    chat_cmd.add_argument("--context", type=int, default=5120)
    chat_cmd.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    cache = Path(os.environ.get("TERNZIP_CACHE", Path.home() / ".cache" / "ternzip")).expanduser().resolve()
    if not dependencies_ready():
        env = cache / "python-3.14-v1"
        python = env / "bin/python"
        with setup_lock(cache):
            if not python.exists():
                venv.EnvBuilder(with_pip=True).create(env)
            if Path(sys.executable) == python:
                subprocess.run([python, "-m", "pip", "--disable-pip-version-check", "install",
                                "--no-input", "--only-binary=:all:", "-r", ROOT / "requirements.txt"],
                               check=True)
        if Path(sys.executable) != python:
            os.execv(python, [str(python), "-B", str(Path(__file__).resolve()), *sys.argv[1:]])
    with setup_lock(cache):
        from native_build import build
        os.environ["TERNZIP_NATIVE"] = str(build(cache))
        if args.command == "chat":
            from metal_backend import install
            os.environ["TERNZIP_BACKEND"] = str(install(cache))
    if args.command in ("restore", "verify"):
        from archive_io import restore, source_name
        output = (args.output or Path(source_name(args.archive))) if args.command == "restore" else None
        report = restore(args.archive, output)
    elif args.command == "pack":
        from encoder import encode
        report = encode(args.gguf, args.output)
    else:
        from chat_model import chat
        report = chat(args.archive, args.prompt, args.context, args.max_tokens)
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ternzip: {error}")
