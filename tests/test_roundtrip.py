"""Round-trip checks: every byte of the Chapter 10 file must be recoverable
from the HDF5 output, and the decoded fields must match the synthetic file.

Run:  python tests/test_roundtrip.py [extra.ch10 ...]
"""

import os
import struct
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from ch10toh5.converter import convert, convert_many  # noqa: E402
import make_sample  # noqa: E402


def check_lossless(ch10_path, h5_path, group="/"):
    """Header fields + body of each packet, plus unparsed bytes, rebuild the file."""
    raw = open(ch10_path, "rb").read()
    covered = np.zeros(len(raw), dtype=bool)
    with h5py.File(h5_path, "r") as h5:
        f = h5[group]["raw"]
        for name, g in f["channels"].items():
            pk = g["packets"][:]
            body = g["body"][:].tobytes()
            for r in pk:
                off, plen = int(r["file_offset"]), int(r["packet_length"])
                hdr = 36 if r["secondary_header"] else 24
                start = off + hdr
                assert raw[start:start + r["data_length"]] == body[r["body_offset"]:r["body_offset"] + r["body_length"]], name
                cid, dtype = struct.unpack_from("<H", raw, off + 2)[0], raw[off + 15]
                assert cid == g.attrs["channel_id"] and dtype == g.attrs["data_type"], name
                assert int.from_bytes(raw[off + 16:off + 22], "little") == r["rtc"], name
                covered[off:off + plen] = True
            if "messages" in g:
                m = g["messages"][:]
                if len(m):
                    assert (m["data_offset"] + m["data_length"] <= len(body)).all(), name
        ub = f["unparsed/bytes"][:].tobytes()
        for o, n, bo in f["unparsed/regions"][:]:
            assert raw[o:o + n] == ub[bo:bo + n]
            covered[o:o + n] = True
        assert f["file_index"].shape[0] == f.attrs["packet_count"]
    assert covered.all(), "bytes not accounted for: %d" % (~covered).sum()


def check_synthetic(h5_path, group="/"):
    exp = make_sample.EXPECT
    with h5py.File(h5_path, "r") as h5:
        f = h5[group]["raw"]
        ch = f["channels"]
        assert f["TMATS/text_000"][()].decode() == exp["tmats"]
        assert dict((k.decode(), v.decode()) for k, v in f["TMATS/attributes"][:])["R-1\\CDT-1"] == "1553IN"

        t = ch["ch0001_TimeF1/messages"][:]
        assert list(t["day_of_year"]) == [100] * 3 and list(t["second"]) == [0, 1, 2]
        assert (np.diff(t["time_ns"]) == 1_000_000_000).all()

        m = ch["ch0002_MIL-STD-1553F1/messages"][:]
        assert len(m) == 6
        assert list(m["rt"][:2]) == [5, 6] and list(m["subaddress"][:2]) == [3, 1]
        assert list(m["rt_to_rt"][:2]) == [0, 1] and m["bus_b"][1] == 1
        assert m["command_word2"][1] >> 11 == 7 and m["gap1"][0] == 3 and m["gap2"][0] == 4
        body = ch["ch0002_MIL-STD-1553F1/body"][:]
        words = body[m["data_offset"][0]:m["data_offset"][0] + m["data_length"][0]].view("<u2")
        assert list(words[1:5]) == [1, 2, 3, 4]
        # 1553 time: 100 RTC ticks (10 us) after the first time packet
        assert m["time_ns"][0] - t["time_ns"][0] == 10_000
        pk = ch["ch0002_MIL-STD-1553F1/packets"][:]
        assert list(pk["decode_status"]) == [0, 0, 0, 2]      # last one is deliberately broken
        assert list(pk["data_checksum_ok"][:3]) == [1, 1, 1]

        pcm = ch["ch0003_PCMF1/messages"][:]
        assert len(pcm) == 15 and (pcm["data_length"] == 20).all() and (pcm["lock_status"] == 15).all()

        a = ch["ch0004_ARINC429F0/messages"][:]
        assert len(a) == 9 and (a["bus"] == 2).all() and list(a["gap_time"][:3]) == [0, 100, 200]
        assert a["label"][0] == 0o203 & 0xFF

        u = ch["ch0005_UARTF0/messages"][:]
        assert (u["ipts_format"] == 2).all() and u["time_ns"][0] == 1_700_000_000_000_001_000
        ubody = ch["ch0005_UARTF0/body"][:].tobytes()
        assert ubody[u["data_offset"][1]:u["data_offset"][1] + u["data_length"][1]] == b"world!"

        e = ch["ch0006_EthernetF0/messages"][:]
        assert list(e["data_length"]) == [60, 61] and (e["ethertype"] == 0x0800).all()
        assert ch["ch0007_MessageF0/messages"]["subchannel"][0] == 9
        assert list(ch["ch0008_DiscreteF1/messages"]["states"]) == [5, 10]
        assert ch["ch0009_AnalogF1/packets"]["csdw_sample_bits"][0] == 16
        assert len(ch["ch0010_VideoF0_MPEG2TS/messages"]) == 2
        c = ch["ch0011_CANBusF0/messages"][0]
        assert c["can_id"] == 0x1ABCDEF and c["extended_id"] == 1 and c["data_length"] == 8
        assert ch["ch0012_Type0x7F/packets"]["csdw"][0] == 0xCAFEF00D
        assert ch["ch0012_Type0x7F/packets"]["data_checksum_ok"][0] == 1
        e1 = ch["ch0013_EthernetF1_ARINC664/messages"][0]
        assert e1["virtual_link"] == 42 and e1["dst_port"] == 5000 and e1["data_length"] == 9
        assert ch["ch0014_IEEE1394F1/packets"]["data_checksum_ok"][0] == 1
        ev = ch["ch0000_ComputerF2_Events/messages"][0]
        assert ev["event_number"] == 7 and ev["event_count"] == 3
        ix = ch["ch0000_ComputerF3_Index/messages"][:]
        assert list(ix["entry_kind"]) == [0, 2] and ix["offset"][0] == 0x100

        regions = f["unparsed/regions"][:]
        assert (regions["file_offset"][0], regions["length"][0]) == exp["junk"]
        assert (regions["file_offset"][1], regions["length"][1]) == exp["tail"]


def main():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "sample.ch10")
    with open(src, "wb") as fh:
        fh.write(make_sample.build())
    for compress in (True, False):
        out = os.path.join(tmp, "sample_%d.h5" % compress)
        convert(src, out, year=2026, compress=compress)
        check_lossless(src, out)
        check_synthetic(out)
    print("synthetic file: OK")

    # Several inputs go into one HDF5 file, one top-level group per input.
    inputs = [src, src] + sys.argv[1:]
    out = os.path.join(tmp, "combined.h5")
    results = convert_many(inputs, out, year=2026)
    groups = [r["group"] for r in results]
    assert groups[:2] == ["/sample", "/sample_2"], groups
    check_synthetic(out, "/sample")
    for path, g in zip(inputs, groups):
        check_lossless(path, out, g)
    with h5py.File(out, "r") as f:
        assert f.attrs["file_count"] == len(inputs)
        assert [x.decode() for x in f["sources"]["group"]] == [g[1:] for g in groups]
    print("combined file (%d inputs): OK" % len(inputs))
    for path in sys.argv[1:]:
        out = os.path.join(tmp, os.path.basename(path) + ".h5")
        res = convert(path, out)
        check_lossless(path, out)
        print("%s: OK (%d packets, %d decode errors)" % (path, res["packets"], res["decode_errors"]))


if __name__ == "__main__":
    main()
