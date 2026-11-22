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


def is_bonsai9(path):
    from versions9 import MAGIC
    with Path(path).open("rb") as stream:
        tag = stream.read(8)
    if tag not in (b"QTile001", MAGIC):
        raise ValueError("Unrecognized Ternzip archive header")
    return tag == MAGIC


def source_name(archive):
    from bonsai9 import SOURCE_NAME as nine_name
    return nine_name if is_bonsai9(archive) else SOURCE_NAME


def restore(archive, destination=None):
    if is_bonsai9(archive):
        import bonsai9
        size, digest = bonsai9.SOURCE_SIZE, bonsai9.SOURCE_SHA
        archive_size, archive_sha = bonsai9.archive_identity(archive)

        def chunks(data):
            return bonsai9.NineArchive(data).original_chunks()
    else:
        from paired_archive import PairImage
        from ternary_codec import original_chunks
        size, digest = SOURCE_SIZE, SOURCE_SHA
        archive_size, archive_sha = ARCHIVE_SIZE, ARCHIVE_SHA

        def chunks(data):
            return original_chunks(PairImage(data))

    with checked_input(archive, archive_size, archive_sha) as (_, data):
        if destination is not None:
            with output_file(destination, size, digest) as out:
                for chunk in chunks(data):
                    if out.write(chunk) != len(chunk):
                        raise OSError("Short write during GGUF restore")
        else:
            count = 0
            checksum = hashlib.sha256()
            for chunk in chunks(data):
                checksum.update(chunk)
                count += len(chunk)
            if count != size or checksum.hexdigest() != digest:
                raise ValueError("Restored GGUF failed the publisher checksum")
    return {"bytes": size, "sha256": digest}
