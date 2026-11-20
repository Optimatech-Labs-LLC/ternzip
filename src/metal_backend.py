"""Install the pinned arm64 Metal libraries for inference."""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request

TAG = "prism-b10828-a153697"
BUNDLE = "llama-" + TAG
URL = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{TAG}/{BUNDLE}-bin-macos-arm64.tar.gz"
TARBALL_SHA = "5c6238a1e7d5c2540ad3d47438388df198433981074290e9397eb4db3d786135"
FILES = {
    "LICENSE": "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d",
    "libggml-base.0.dylib": "52043e054006f041d0bd8e1da8e4e3db3b8be7796451cd57b3c60248c9b2cd6c",
    "libggml-blas.0.dylib": "a0d2f7baa25dff0439d765f04af345e6f769111e1eaa392c32a689b17b5e2690",
    "libggml-cpu.0.dylib": "6decb0f90736d73fb63978652182ff955ef958487342977bec1cc80716b82586",
    "libggml-metal.0.dylib": "1749f34a6ae48143ae046bccc08650c3881485e97d6125b3739760a810d86c31",
    "libggml-rpc.0.dylib": "6bd0c1e9251f2c56402d6f6b689de3cee93e655e0bb17eb9dc36e3e42e685b6c",
    "libggml.0.dylib": "aa84e1ce5256bf9bce26151455f25265ed50144db501e9c805bf0ce827d54319",
    "libllama.0.dylib": "b0645436f65c5eee2e697a1313e4219052706691fdc967fd219120db715e0bfc",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(cache):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("Chat requires Apple Silicon and Metal.")
    root = cache / "backend"
    root.mkdir(parents=True, exist_ok=True)
    target = root / TAG
    if target.is_dir():
        if all((target / name).is_file() and digest(target / name) == sha for name, sha in FILES.items()):
            return target
        raise ValueError(f"Cached backend checksum mismatch in {target}. Move that directory and retry.")
    with tempfile.TemporaryDirectory(dir=root) as temp:
        stage = Path(temp)
        source = os.environ.get("TERNZIP_BACKEND_ARCHIVE")
        archive = Path(source) if source else stage / "runtime.tar.gz"
        if not source:
            with urllib.request.urlopen(URL, timeout=90) as response, archive.open("xb") as out:
                shutil.copyfileobj(response, out)
        if digest(archive) != TARBALL_SHA:
            raise ValueError("Published backend archive SHA-256 mismatch")
        libraries = stage / "libraries"
        libraries.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            for name, sha in FILES.items():
                path = libraries / name
                with tar.extractfile(f"{BUNDLE}/{name}") as stream, path.open("xb") as out:
                    shutil.copyfileobj(stream, out)
                if digest(path) != sha:
                    raise ValueError(f"Backend file SHA-256 mismatch for {name}")
        libraries.rename(target)
    return target
