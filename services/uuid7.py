"""RFC 9562 UUID version 7 generation.

UUIDv7 embeds a 48-bit millisecond Unix timestamp in the high bits, so IDs
generated later sort after IDs generated earlier -- useful for a permanent
document_id where insertion order happens to matter (e.g. for debugging/
inspection), while still being a standard 128-bit UUID everywhere else.

This is a small, dependency-free implementation (no `uuid7`/`uuid6` PyPI
package required) since Python's own `uuid` module only gained `uuid.uuid7()`
in 3.14. If/when this project's minimum Python version reaches 3.14, this
module can be replaced with a thin wrapper around the stdlib version without
changing any caller.
"""

import os
import time
import uuid


def uuid7() -> uuid.UUID:
    """Generate one RFC 9562 UUIDv7 value.

    Layout (128 bits total):
      - bits 127-80 (48 bits): unix_ts_ms  -- milliseconds since the epoch
      - bits 79-76  (4 bits):  version     -- always 0b0111 (7)
      - bits 75-64  (12 bits): rand_a      -- random
      - bits 63-62  (2 bits):  variant     -- always 0b10
      - bits 61-0   (62 bits): rand_b      -- random
    """
    unix_ts_ms = time.time_ns() // 1_000_000

    # 10 random bytes (80 bits) is enough to cover rand_a (12 bits) +
    # rand_b (62 bits) = 74 bits needed; the extra bits are simply masked off.
    rand = os.urandom(10)
    rand_a = int.from_bytes(rand[0:2], "big") & 0x0FFF
    rand_b = int.from_bytes(rand[2:10], "big") & 0x3FFF_FFFF_FFFF_FFFF

    uuid_int = (unix_ts_ms & 0xFFFF_FFFF_FFFF) << 80
    uuid_int |= 0x7 << 76
    uuid_int |= rand_a << 64
    uuid_int |= 0b10 << 62
    uuid_int |= rand_b

    return uuid.UUID(int=uuid_int)


def uuid7_str() -> str:
    """Convenience wrapper returning the canonical 36-character string form
    (e.g. "018f1e2a-7c3d-7e4a-9b1a-2f3e4d5c6b7a"), which is what gets stored
    as document_id everywhere in this project."""
    return str(uuid7())