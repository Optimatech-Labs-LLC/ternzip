"""Runtime locations resolved by the public launcher."""
import os
from pathlib import Path
import sys


def library(name):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return Path(os.environ["TERNZIP_NATIVE"]) / ("lib" + name + suffix)


def backend_directory():
    return Path(os.environ["TERNZIP_BACKEND"])
