"""Stand-in for the PyPI package "zstd", which asammdf imports unconditionally.

`asammdf/blocks/utils.py` and `asammdf/blocks/v4_blocks.py` do, at module
load time::

    from zstd import decompress as zstd_decompress
    from zstd import compress as zstd_compress

That is the project literally named "zstd" on PyPI (not "zstandard", a
different, actively maintained project). "zstd" stopped publishing Windows
wheels once Python passed 3.10: its most recent release ships cp35-cp310
wheels for win32/win_amd64 and nothing newer. On Python 3.11+ on Windows,
`pip install zstd` (or `pip install asammdf`, which pulls it in) falls back
to building the C extension from source, which needs Microsoft's C++ Build
Tools -- rarely present on a measurement PC -- and fails with exactly the
"Microsoft Visual C++ 14.0 or greater is required" error this exists to
route around.

asammdf's block reader/writer only ever calls these with the buffer to
convert and, for `compress`, a level; it never touches anything else in the
"zstd" namespace, and this pipeline never calls `compress` at all -- it only
reads bench logs, never writes MDF4 -- but the import of it still runs at
module load regardless of what gets called, so both have to exist here or
`import asammdf` fails before either is ever used. `zstandard.decompress`
and `zstandard.compress` are exactly those two functions, and `zstandard`
DOES ship Windows wheels for every current CPython version, because it is
maintained and `zstd` is not. So rather than asking the user's machine to
compile a C extension it was never going to need for reading their own
instrument's files (Dewetron / GAMRY logs use uncompressed or LZ4 blocks in
every file this pipeline has been run against; a ZSTD-compressed block
would still decompress correctly through this shim, it just hasn't come
up), this module satisfies the import with the equivalent calls into
`zstandard` instead.

If the real "zstd" package IS installed (Linux/macOS wheels still exist, or
someone has a working compiler), Python's normal import resolution finds it
before this shim only if this vendor directory is NOT put ahead of it on
sys.path -- see `_ensure_zstd_importable()` in gamry_compare.py, which
checks for the real package first and only falls back to this one.
"""

from __future__ import annotations

# noqa: F401 on both -- re-exported so `from zstd import decompress` and
# `from zstd import compress` resolve the way asammdf expects.
from zstandard import compress, decompress  # noqa: F401
