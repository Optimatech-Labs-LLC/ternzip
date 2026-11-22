# Apple Silicon runtime for Ternzip 9B

Ternzip v0.3.1 bundles the arm64 backend for ordinary single-stream chat. It preserves the model’s ternary weights and 16-bit block scales. Ternzip stores the Hadamard rotation exactly in a compact representation, reuses single-token graph operations and evaluates the four-tap convolution in parallel across channels. The execution layout uses two-bit slots for 86 feed-forward matrices. The rest of the ternary compute matrices use dense trits.

The five patches in this directory apply to the same Prism source revision. The C++ changes cover the direct loading, rotation, recurrent state and Qwen3.5 graph. The Metal changes cover the single-state write and the channel convolution. Apply them from the root of a clean upstream checkout with `patch -p1 < path/to/patch` for each file. The arm64 bundle contains two modified inference libraries, five upstream ggml libraries and the fast-wait helper. The runtime carries the MIT license. [Pinned Prism source](https://github.com/PrismML-Eng/llama.cpp/tree/a15369706017f177cae8aa40b6b8592259a48806) · [Checksums used by the launcher](../src/metal9.json) · [License](LICENSE)

The launcher fetches the versioned bundle and verifies its SHA-256 before installing it. For an offline setup, set `TERNZIP_9B_BACKEND_ARCHIVE` to a local copy of `ternzip-metal-macos-arm64-v0.3.1.tar.gz`. It verifies both the bundle and each extracted library. The restore command works without Metal.

## Rebuild on Apple Silicon

Start in an empty directory on an Apple Silicon Mac with Git, CMake and Apple’s Command Line Tools or Xcode. These commands compile the v0.3.1 backend from Prism’s pinned source. They include the Metal kernels in the shared library and target macOS 14 or later.

```sh
git clone --depth 1 --branch v0.3.1 https://github.com/Optimatech-Labs-LLC/ternzip.git ternzip
git init --quiet prism-llama
git -C prism-llama fetch --depth 1 https://github.com/PrismML-Eng/llama.cpp.git a15369706017f177cae8aa40b6b8592259a48806
git -C prism-llama checkout --quiet --detach FETCH_HEAD

patch_dir="$PWD/ternzip/backend"
git -C prism-llama apply "$patch_dir/rotation.patch" "$patch_dir/qwen35.patch" \
  "$patch_dir/delta-net.patch" "$patch_dir/metal.patch" "$patch_dir/direct-load.patch"

cmake -S prism-llama -B build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_BLAS=ON -DGGML_RPC=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=@loader_path' \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
cmake --build build --config Release --target llama --parallel 6
```

The seven arm64 libraries are under `build/bin`. From the same directory, this check loads the Metal and model libraries and checks the direct writer and calls the rotation and graph entry points. It requires Python 3 and no model download.

```sh
python3 - <<'PY'
import ctypes as C
from pathlib import Path

libdir = Path("build/bin").resolve()
for name in ("ggml-base", "ggml-blas", "ggml-cpu", "ggml-metal", "ggml-rpc", "ggml", "llama"):
    assert (libdir / f"lib{name}.0.dylib").is_file(), name
C.CDLL(str(libdir / "libggml-metal.0.dylib"))
llama = C.CDLL(str(libdir / "libllama.0.dylib"))
assert hasattr(llama, "ternzip9_load_with_writer")
llama.ternzip9_reuse_config.argtypes = [C.c_uint]
llama.ternzip9_reuse_config.restype = None
llama.ternzip9_reuse_config(8)
llama.ternzip9_rotation_format.argtypes = [C.c_uint32]
llama.ternzip9_rotation_format.restype = C.c_int
assert llama.ternzip9_rotation_format(4096) == 41  # GGML_TYPE_Q1_0 in the pinned source
assert llama.ternzip9_rotation_format(1024) == 0   # GGML_TYPE_F32
print("Ternzip backend loaded successfully")
PY
```

To test a full chat with your own build, use CPython 3.14, stable Rust and the 9B archive. The following development command compiles the four codec libraries from the tagged Ternzip source, loads the seven libraries from `build/bin`, and checks the model archive through the usual 9B decoding path. Run it from the directory containing `build` and `ternzip`.

```sh
python3.14 -m venv .ternzip-dev
.ternzip-dev/bin/python -m pip install -r ternzip/requirements.txt
.ternzip-dev/bin/python - /path/to/ternzip-9b.trits <<'PY'
import os
from pathlib import Path
import sys

work = Path.cwd()
sys.path.insert(0, str(work / "ternzip/src"))
from native_build import build

os.environ["TERNZIP_NATIVE"] = str(build(work / "build/codecs"))
os.environ["TERNZIP_BACKEND"] = str(work / "build/bin")
from chat_model import chat

reply = chat(Path(sys.argv[1]), "What is 17 × 6? Answer with the number only.", 5120, 32)
print(reply["devices"], reply["text"], reply["stop"])
PY
```

The release targets the same pinned Prism revision as v0.3.0. The additional patch exposes the direct loader for final Metal buffers.

This is a development build. Compiler versions, linker inputs and code signatures affect its binary hashes. The distributed release reused five Prism libraries and supplied two modified libraries. The public launcher checks the SHA-256 of that specific release archive and every library inside it. The local build stays in `build/bin`. [Release checksum manifest](../src/metal9.json)

## Rebuild the optional fast-wait helper

From the directory containing the pinned Prism source in `prism-llama` and Ternzip in `ternzip`, build the companion next to the seven inference libraries. It submits an empty completion marker on the registered Metal device's command queue and polls it during single-token generation. The original token read and backend error handling still run.

```sh
clang -std=c11 -O2 -arch arm64 -mmacosx-version-min=14.0 -dynamiclib \
  -Iprism-llama/ggml/include -Iprism-llama/ggml/src ternzip/backend/metal-wait.c \
  -lobjc -Wl,-install_name,@rpath/libternzip-wait.dylib -o build/bin/libternzip-wait.dylib
codesign --force --sign - build/bin/libternzip-wait.dylib
```
