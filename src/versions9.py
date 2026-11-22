"""Pinned on-disk model identities supported by the 9B loader."""
from pathlib import Path

MAGIC = b"QBons9R1"
SOURCE_NAME = "Ternary-Bonsai-2-9B-PTQ1_0.gguf"
SOURCE_SIZE = 1_997_486_976
SOURCE_SHA = "3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b"
ARCHIVE_SIZE = 1_834_157_871
ARCHIVE_SHA = "47f52b0d8e0ec4d38f61d61e7100ec3b17fec6dbe2fe58a7904fa4a57c07a66b"
ARCHIVES = {
    ARCHIVE_SIZE: ("v0.3.0", ARCHIVE_SHA),
    1_860_057_576: ("v0.2.0", "a53379d2406c3be10498475e9722818986e534e3bef5cd4690b35ae74bdbd40f"),
}


def archive_identity(path):
    size = Path(path).stat().st_size
    if size not in ARCHIVES:
        raise ValueError(
            "Unsupported 9B archive version or incomplete download. This loader is v0.3.1 "
            "and supports the v0.2.0 and v0.3.0 archives. Check SHA256SUMS and use the "
            "loader tag linked by the archive's model card."
        )
    return size, ARCHIVES[size][1]
