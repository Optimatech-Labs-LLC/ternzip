"""Pinned model identities and atomic verified output publication."""
from contextlib import contextmanager
import errno
import hashlib
import mmap
import os
from pathlib import Path
import tempfile

SOURCE_NAME = "Ternary-Bonsai-8B-Q2_0.gguf"
SOURCE_SIZE = 2_182_184_672
SOURCE_SHA = "3c8d70470a5d97e5a2b9410ddd899cb740116591462626c60cb2fead6448f60b"
ARCHIVE_NAME = "ternzip-8b.trits"
ARCHIVE_SIZE = 1_623_736_952
ARCHIVE_SHA = "19ce2237a410208327f69a9f5eff381d2d6e8ffe9dd750f65a433e7f3c747577"


@contextmanager
def checked_input(path, size, expected):
    with Path(path).open("rb") as src:
        if os.fstat(src.fileno()).st_size != size:
            raise ValueError(f"Unexpected byte size for {path}. Expected {size:,}.")
        with mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch for {path}")
            yield src, data


def verify_file(path, size, expected):
    with Path(path).open("rb") as src:
        if os.fstat(src.fileno()).st_size != size or hashlib.file_digest(src, "sha256").hexdigest() != expected:
            raise ValueError(f"Size or SHA-256 mismatch for {path}")


@contextmanager
def output_file(destination, size, expected):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".ternzip-", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with temp.open("r+b") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        verify_file(temp, size, expected)
        try:
            os.link(temp, destination)
        except OSError as exc:
            if exc.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM):
                raise
            with destination.open("xb") as reserved:
                try:
                    os.replace(temp, destination)
                except BaseException:
                    if os.path.samestat(os.fstat(reserved.fileno()), destination.stat()):
                        destination.unlink()
                    raise
    finally:
        temp.unlink(missing_ok=True)


def restore(archive, destination=None):
    from paired_archive import PairImage
    from ternary_codec import original_chunks
    with checked_input(archive, ARCHIVE_SIZE, ARCHIVE_SHA) as (_, data):
        image = PairImage(data)
        if destination is not None:
            with output_file(destination, SOURCE_SIZE, SOURCE_SHA) as out:
                for chunk in original_chunks(image):
                    out.write(chunk)
        else:
            size = 0
            digest = hashlib.sha256()
            for chunk in original_chunks(image):
                digest.update(chunk)
                size += len(chunk)
            if size != SOURCE_SIZE or digest.hexdigest() != SOURCE_SHA:
                raise ValueError("Restored GGUF failed the publisher checksum")
    return {"bytes": SOURCE_SIZE, "sha256": SOURCE_SHA}
