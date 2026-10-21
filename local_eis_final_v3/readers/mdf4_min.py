"""
mdf4_min.py
===========
A minimal MDF 4.x reader for the small, sorted files the Gamry export writes
(RO2612090-01_V26_112_HFR_101_CurrVal_450.mf4: one data group, one channel
group, float64/uint channels, an XML header comment with STARTTIME).

Exists because `asammdf` is often not installed on the cluster, and a Gamry
sweep saved as .mf4 must still be readable as the whole-cell reference.
Handles ##DT, ##DL and ##DZ (deflate, with or without transposition) data,
little/big-endian integers and floats, and linear conversions. Anything
beyond that (VLSD, bus logging, bit fields across bytes) raises.
"""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path

import numpy as np


class MDF4Error(ValueError):
    pass


def _block(buf: bytes, at: int):
    """(id, links, data bytes) of the block at byte `at`."""
    if at <= 0 or at + 24 > len(buf):
        raise MDF4Error(f"block link {at} outside the file")
    bid = buf[at:at + 4].decode("latin-1")
    if not bid.startswith("##"):
        raise MDF4Error(f"no block at {at} ({bid!r})")
    length, n_links = struct.unpack_from("<QQ", buf, at + 8)
    links = list(struct.unpack_from(f"<{n_links}Q", buf, at + 24))
    data = buf[at + 24 + 8 * n_links: at + length]
    return bid[2:], links, data


def _text(buf: bytes, at: int) -> str:
    if not at:
        return ""
    _, _, data = _block(buf, at)
    return data.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _data_bytes(buf: bytes, at: int) -> bytes:
    """Concatenated record bytes of a ##DT / ##DL / ##DZ chain."""
    if not at:
        return b""
    bid, links, data = _block(buf, at)
    if bid in ("DT", "DV"):
        return data
    if bid == "DZ":
        org = data[0:2].decode("latin-1")
        ztype = data[2]
        zparam, org_len, z_len = struct.unpack_from("<IQQ", data, 4)
        raw = zlib.decompress(data[24:24 + z_len])
        if ztype == 1 and zparam > 1:                     # transposed
            cols = zparam
            rows = len(raw) // cols
            main = np.frombuffer(raw[:rows * cols], np.uint8
                                 ).reshape(cols, rows).T.tobytes()
            raw = main + raw[rows * cols:]
        if org not in ("DT", "DV"):
            raise MDF4Error(f"##DZ of {org} not supported")
        return raw[:org_len]
    if bid == "DL":
        out = []
        while True:
            for link in links[1:]:
                if link:
                    out.append(_data_bytes(buf, link))
            if not links[0]:
                break
            bid, links, data = _block(buf, links[0])
        return b"".join(out)
    if bid == "HL":
        return _data_bytes(buf, links[0])
    raise MDF4Error(f"data block ##{bid} not supported")


_DTYPES = {0: "<u", 1: ">u", 2: "<i", 3: ">i", 4: "<f", 5: ">f"}


def read(path) -> tuple[dict, dict]:
    """({channel name: values}, {header key: value}) of a sorted MDF4 file.

    Header keys come from the <e name="..."> elements of the header comment
    (STARTTIME, measurement_start, IACREQ, ...) plus 'start_time_ns'."""
    buf = Path(path).read_bytes()
    if not buf.startswith(b"MDF "):
        raise MDF4Error(f"{Path(path).name}: not an MDF file")
    _, hd_links, hd_data = _block(buf, 64)
    meta = {"start_time_ns": struct.unpack_from("<Q", hd_data, 0)[0]}
    comment = _text(buf, hd_links[5]) if len(hd_links) > 5 else ""
    for k, v in re.findall(r'<e name="([^"]+)"[^>]*>([^<]*)</e>', comment):
        meta[k] = v.strip()

    chans: dict[str, np.ndarray] = {}
    dg = hd_links[0]
    while dg:
        _, dg_links, dg_data = _block(buf, dg)
        rec_id_size = dg_data[0] if dg_data else 0
        records = _data_bytes(buf, dg_links[2])
        cg = dg_links[1]
        n_cg = 0
        while cg:
            n_cg += 1
            _, cg_links, cg_data = _block(buf, cg)
            cycles = struct.unpack_from("<Q", cg_data, 8)[0]
            data_bytes, inval_bytes = struct.unpack_from("<II", cg_data, 24)
            stride = rec_id_size + data_bytes + inval_bytes
            if n_cg > 1:
                raise MDF4Error("unsorted MDF (several channel groups in one "
                                "data group) not supported")
            n = min(cycles, len(records) // stride if stride else 0)
            rec = np.frombuffer(records[:n * stride], np.uint8).reshape(n, stride)
            cn = cg_links[1]
            while cn:
                _, cn_links, cn_data = _block(buf, cn)
                name = _text(buf, cn_links[2])
                (cn_type, _sync, dtype, bit_off, byte_off,
                 bit_count) = struct.unpack_from("<BBBBII", cn_data, 0)
                if dtype not in _DTYPES or bit_off or bit_count % 8:
                    raise MDF4Error(f"channel {name}: data type {dtype}, "
                                    f"{bit_count} bits not supported")
                nb = bit_count // 8
                a = rec_id_size + byte_off
                raw = np.ascontiguousarray(rec[:, a:a + nb])
                vals = raw.view(f"{_DTYPES[dtype]}{nb}").ravel().astype(float)
                cc = cn_links[4]
                if cc:
                    _, _, cc_data = _block(buf, cc)
                    cc_type = cc_data[0]
                    val_count = struct.unpack_from("<H", cc_data, 6)[0]
                    p = struct.unpack_from(f"<{val_count}d", cc_data, 24)
                    if cc_type == 1 and len(p) >= 2:
                        vals = p[0] + p[1] * vals
                    elif cc_type != 0:
                        raise MDF4Error(f"channel {name}: conversion type "
                                        f"{cc_type} not supported")
                chans[name] = vals
                cn = cn_links[0]
            cg = cg_links[0]
        dg = dg_links[0]
    return chans, meta
