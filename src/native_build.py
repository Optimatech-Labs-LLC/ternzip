"""Build the C-ABI codec libraries from the included Rust sources."""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parent
MODULES = {
    "trit-rans": ("trit_rans.rs", ("trit_encode", "trit_decode")),
    "quota-rans": ("quota_rans.rs", ("quota_decode",)),
    "word-rans": ("word_rans.rs", ("word_decode",)),
    "load-kernels": ("load_kernels.rs", ("pack_native", "corrected_intervals", "row_intervals",
                     "block_intervals", "index_stream", "seek_row", "inverse_embedding")),
}


def build(cache):
    stamp = hashlib.sha256(Path(__file__).read_bytes())
    for filename, _ in MODULES.values():
        stamp.update((SOURCE / filename).read_bytes())
    mac = sys.platform == "darwin"
    suffix = ".dylib" if mac else ".so"
    tag = f"{sys.platform}-{platform.machine()}-{stamp.hexdigest()[:16]}"
    dest = cache / "native" / tag
    if all((dest / ("lib" + name + suffix)).is_file() for name in MODULES):
        return dest
    rust = shutil.which("rustc")
    if rust is None:
        raise RuntimeError("Install the stable Rust toolchain (rustc), then rerun this command.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    libdir = Path(subprocess.check_output([rust, "--print", "target-libdir"], text=True).strip())
    linker = libdir.parent / "bin/gcc-ld/ld64.lld"
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        here = Path(temp)
        built = here / "out"
        built.mkdir()
        for name, (filename, symbols) in MODULES.items():
            lib = built / ("lib" + name + suffix)
            flags = [rust, "--edition=2021", "-C", "opt-level=3", "-C", "panic=abort",
                     "--remap-path-prefix", f"{SOURCE}=src"]
            if mac and platform.machine() == "arm64" and linker.is_file():
                obj = here / (name + ".o")
                subprocess.run([*flags, "--crate-type=lib", "--emit=obj", filename, "-o", obj], cwd=SOURCE, check=True)
                archives = [next(libdir.glob("lib" + item + "-*.rlib")) for item in ("core", "compiler_builtins")]
                exports = [part for symbol in symbols for part in ("-exported_symbol", "_" + symbol)]
                subprocess.run([linker, "-arch", "arm64", "-dylib", "-platform_version", "macos", "11.0",
                                "11.0", "-install_name", "@rpath/" + lib.name, "-dead_strip", *exports,
                                "-undefined", "dynamic_lookup", "-o", lib, obj, *archives], check=True)
            else:
                subprocess.run([*flags, "--crate-type=cdylib", "-C", "strip=debuginfo",
                                filename, "-o", lib], cwd=SOURCE, check=True)
        os.rename(built, dest)
    return dest
