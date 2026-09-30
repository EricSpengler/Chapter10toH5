"""Checks for engineering-unit measurements (CSV and TMATS definitions).

Run:  python tests/test_measurements.py
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
from make_sample import RTC_HZ, Writer, bcd_time_doy  # noqa: E402

CSV = """# name,type,... (comment lines are ignored)
name,type,channel,rt,tr,sa,word,words,label,sdi,bus,lsb,bits,encoding,scale,offset,coefficients,units,description
half_word2,1553,,5,0,3,2,,,,,,,,0.5,,,V,RT5 SA3 data word 2 scaled by 0.5
rtrt_long,1553,,7,1,1,1,2,,,,,,,,,,counts,RT-RT transmit side two words
rtrt_signed,1553,,7,,1,1,,,,,,16,signed,,,,counts,
arinc_214,arinc429,,,,,,,214,,2,,,,2,10,,ft,
arinc_poly,arinc429,4,,,,,,214,,,,,,,,1;0;1,x,1 + x^2
bad row,1553,,,,,,,,,,,,,,,,,
"""


def pcm_file():
    """PCM with intra-packet headers, 5-word frames, and TMATS D/C definitions."""
    tmats = "\n".join([
        "G\\PN:PCM demo;", "R-1\\TK1-1:3;", "R-1\\DSI-1:PCM1;", "R-1\\CDT-1:PCMIN;",
        "P-1\\DLN:PCM1;", "P-1\\D2:96000;", "P-1\\F1:16;", "P-1\\MF1:5;", "P-1\\MF2:96;", "P-1\\MF4:32;",
        "P-1\\MF5:11111110011010110010100001000000;",
        "D-1\\DLN:PCM1;", "D-1\\MN-1-1:TEMP;", "D-1\\LT-1-1:WDFR;", "D-1\\WP-1-1-1:3;",
        "D-1\\MN-1-2:PRESS;", "D-1\\LT-1-2:WDFR;", "D-1\\WP-1-2-1:2;", "D-1\\WI-1-2-1:2;",
        "D-1\\MN-1-3:STATUS_HI;", "D-1\\WP-1-3-1:5;", "D-1\\WFM-1-3-1-1:1111000000000000;",
        "C-1\\DCN:TEMP;", "C-1\\MN4:degC;", "C-1\\BFM:UNS;", "C-1\\DCT:COE;", "C-1\\CO\\N:1;",
        "C-1\\CO:-40;", "C-1\\CO-1:0.5;",
        "C-2\\DCN:PRESS;", "C-2\\MN4:kPa;", "C-2\\DCT:PTS;", "C-2\\PS\\N:2;",
        "C-2\\PS1-1:0;", "C-2\\PS2-1:0;", "C-2\\PS1-2:1000;", "C-2\\PS2-2:100;", ""])
    w = Writer()
    rtc0 = 0x2000_0000
    w.packet(0, 0x01, struct.pack("<I", 7) + tmats.encode(), rtc0)
    w.packet(1, 0x11, struct.pack("<I", 0x11) + bcd_time_doy(10, 0, 0, 0, 0), rtc0)
    expect = []
    for p in range(4):
        body = b""
        for k in range(5):
            n = p * 5 + k
            ipts = rtc0 + n * 10_000
            words = [0xFE6B, 0x2840, n * 10, 100 + n, n * 10 + 5, 0xA000 | n]
            body += struct.pack("<QH", ipts, 0xF000) + struct.pack("<6H", *words)
            expect.append(n)
        csdw = (1 << 30) | (1 << 19) | (3 << 24) | (3 << 26)
        w.packet(3, 0x09, struct.pack("<I", csdw) + body, rtc0 + p * RTC_HZ // 100)
    return bytes(w.out), np.array(expect)


def bus_file():
    """RT 9 SA 2 word 1 = sample number, recorded three ways:
    channel 2 (seconds 0-9), channel 3 (the same bus again, 2 us later) and
    channel 4 (seconds 10-19, a different part of the recording)."""
    w = Writer()
    rtc0 = 0x3000_0000
    w.packet(0, 0x01, struct.pack("<I", 7) + b"G\\PN:bus demo;\nR-1\\TK1-1:4;\nR-1\\DSI-1:BUS-B;\n", rtc0)
    w.packet(1, 0x11, struct.pack("<I", 0x11) + bcd_time_doy(10, 0, 0, 0, 0), rtc0)
    cw = (9 << 11) | (1 << 10) | (2 << 5) | 1
    for ch, secs, shift in ((2, range(0, 10), 0), (3, range(0, 10), 20), (4, range(10, 20), 0)):
        body = b""
        for k in secs:
            for j in range(10):                     # 10 Hz
                n = k * 10 + j
                data = struct.pack("<3H", cw, 9 << 11, n)
                body += struct.pack("<QHHH", rtc0 + n * RTC_HZ // 10 + shift, 0, 0, len(data)) + data
        w.packet(ch, 0x19, struct.pack("<I", len(secs) * 10) + body, rtc0)
    return bytes(w.out)


def commutated_file(with_counter=True):
    """PCM with a 4-minor-frame major frame: a subframe ID counter in word 2, a supercommutated
    measurement in words 3 and 5, and a subcommutated word 4 whose meaning changes each minor frame.
    Minor frames arrive every 1 ms (1000 Hz)."""
    tm = ["G\\PN:commutation demo;", "R-1\\TK1-1:3;", "R-1\\DSI-1:PCM1;", "R-1\\CDT-1:PCMIN;",
          "P-1\\DLN:PCM1;", "P-1\\D2:1000000;", "P-1\\F1:16;", "P-1\\MF1:6;", "P-1\\MF2:112;", "P-1\\MF4:32;",
          "P-1\\MF5:11111110011010110010100001000000;", "P-1\\MF\\N:4;"]
    if with_counter:
        tm += ["P-1\\ISF\\N:1;", "P-1\\ISF2-1:ID;", "P-1\\IDC1-1:2;", "P-1\\IDC3-1:15;", "P-1\\IDC4-1:2;",
               "P-1\\IDC6-1:0;", "P-1\\IDC10-1:INC;"]
    tm += ["D-1\\DLN:PCM1;",
           # supercommutated: words 3 and 5 of every minor frame (TMATS 5-index form)
           "D-1\\MN-1-1:SUPER;", "D-1\\LT-1-1:WDFR;", "D-1\\WP-1-1-1-1:3;", "D-1\\WI-1-1-1-1:2;",
           "D-1\\EWP-1-1-1-1:5;",
           # subcommutated: word 4 of minor frame 2 only
           "D-1\\MN-1-2:SUB2;", "D-1\\LT-1-2:WDFR;", "D-1\\WP-1-2-1:4;", "D-1\\FP-1-2-1:2;",
           # subcommutated: word 4 of minor frames 1 and 3
           "D-1\\MN-1-3:SUB13;", "D-1\\LT-1-3:WDFR;", "D-1\\WP-1-3-1:4;", "D-1\\FP-1-3-1:1;",
           "D-1\\FI-1-3-1:2;",
           # the same data as SUPER, listed as two separate locations
           "D-1\\MN-1-4:MULTI;", "D-1\\LT-1-4:WDFR;", "D-1\\MML\\N-1-4:2;", "D-1\\WP-1-4-1:3;",
           "D-1\\WP-1-4-2:5;", ""]
    w = Writer()
    rtc0 = 0x2000_0000
    w.packet(0, 0x01, struct.pack("<I", 7) + "\n".join(tm).encode(), rtc0)
    w.packet(1, 0x11, struct.pack("<I", 0x11) + bcd_time_doy(10, 0, 0, 0, 0), rtc0)
    per_packet, n_frames = 4, 40
    for p in range(n_frames // per_packet):
        body = b""
        for k in range(per_packet):
            n = p * per_packet + k
            frame = n % 4 + 1
            words = [0xFE6B, 0x2840, 0xAB00 | (n % 4), 1000 + n, 100 * frame + n, 2000 + n, 7]
            body += struct.pack("<QH", rtc0 + n * RTC_HZ // 1000, 0xF000) + struct.pack("<7H", *words)
        csdw = (1 << 30) | (1 << 19) | (3 << 24) | (3 << 26)
        w.packet(3, 0x09, struct.pack("<I", csdw) + body, rtc0 + p * per_packet * RTC_HZ // 1000)
    return bytes(w.out), np.arange(n_frames)


def check_commutation(tmp):
    from ch10toh5.measurements import observed_rate
    data, n = commutated_file()
    src = os.path.join(tmp, "commutated.ch10")
    with open(src, "wb") as fh:
        fh.write(data)
    defs = os.path.join(tmp, "commutated.csv")
    with open(defs, "w") as fh:
        fh.write("name,type,channel,word,frame,frame_interval\n"
                 "CSV_SUB3,pcm,,4,3,\n"          # word 4 of minor frame 3, counter from TMATS
                 "CSV_EVERY,pcm,,4,,\n")         # no frame given: word 4 of every minor frame
    out = os.path.join(tmp, "commutated.h5")
    logs = []
    convert(src, out, year=2026, definitions=defs, log=logs.append)
    with h5py.File(out, "r") as f:
        frame_rate = observed_rate(f["raw/channels/ch0003_PCMF1/messages"]["time_ns"][:])
        assert abs(frame_rate - 1000) < 1, frame_rate
        sup = f["SUPER"]
        assert sorted(sup["raw"][:]) == sorted(list(1000 + n) + list(2000 + n))
        assert len(sup["raw"]) == 2 * len(n)
        assert list(f["SUB2/raw"][:]) == [200 + i for i in n if i % 4 == 1]
        assert abs(f["SUB2"].attrs["rate_hz_observed"] - 250) < 1
        # supercommutated: twice the minor frame rate; subcommutated: a quarter or half of it
        assert abs(sup.attrs["rate_hz_observed"] - 2000) < 30, sup.attrs["rate_hz_observed"]
        assert abs(f["SUB13"].attrs["rate_hz_observed"] - 500) < 10
        assert abs(f["CSV_SUB3"].attrs["rate_hz_observed"] - 250) < 5
        assert list(f["SUB13/raw"][:]) == [100 * (i % 4 + 1) + i for i in n if i % 4 in (0, 2)]
        assert list(f["CSV_SUB3/raw"][:]) == [300 + i for i in n if i % 4 == 2]
        assert len(f["CSV_EVERY/raw"]) == len(n)
        assert sorted(f["MULTI/raw"][:]) == sorted(sup["raw"][:])
    print("PCM supercommutation and subcommutation from TMATS and CSV: OK")

    # no counter in the TMATS: subcommutated words are skipped unless the CSV names the counter
    data, n = commutated_file(with_counter=False)
    src = os.path.join(tmp, "nocounter.ch10")
    with open(src, "wb") as fh:
        fh.write(data)
    with open(defs, "w") as fh:
        fh.write("name,type,channel,word,frame,sfid_word,sfid_lsb,sfid_bits,frames\n"
                 "CSV_SUB4,pcm,3,4,4,2,0,2,4\n"
                 "CSV_NOCOUNTER,pcm,3,4,4,,,,\n")
    out = os.path.join(tmp, "nocounter.h5")
    logs = []
    convert(src, out, year=2026, definitions=defs, log=logs.append)
    with h5py.File(out, "r") as f:
        assert list(f["CSV_SUB4/raw"][:]) == [400 + i for i in n if i % 4 == 3]
        assert len(f["CSV_NOCOUNTER/raw"]) == 0 and len(f["SUB2/raw"]) == 0
        assert len(f["SUPER/raw"]) == 2 * len(n)
    assert any("CSV_NOCOUNTER" in m and "subframe ID counter" in m for m in logs), logs
    print("subframe ID counter from the CSV, and skip without one: OK")


def main():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "sample.ch10")
    with open(src, "wb") as fh:
        fh.write(make_sample.build())
    defs = os.path.join(tmp, "defs.csv")
    with open(defs, "w") as fh:
        fh.write(CSV)
    out = os.path.join(tmp, "sample.h5")
    logs = []
    convert(src, out, year=2026, definitions=defs, log=logs.append)
    assert any("line 8" in m for m in logs), logs          # the bad row is reported
    with h5py.File(out, "r") as f:
        assert {"raw", "half_word2", "measurement_index"} <= set(f)
        g = f["half_word2"]
        assert list(g["raw"][:]) == [2, 2, 2] and list(g["value"][:]) == [1.0, 1.0, 1.0]
        assert g.attrs["units"] == "V"
        t1553 = f["raw/channels/ch0002_MIL-STD-1553F1/messages"]["time_ns"][:]
        assert list(g["time_ns"][:]) == list(t1553[[0, 2, 4]])
        assert list(g.attrs["sources"]) == ["ch0002_RT5_R_SA3_W2"]
        assert list(f["rtrt_long/raw"][:]) == [0xAAAA5555] * 3
        assert list(f["rtrt_signed/value"][:]) == [0xAAAA - 0x10000] * 3
        assert list(f["arinc_214/raw"][:]) == [1, 2] * 3
        assert list(f["arinc_214/value"][:]) == [12.0, 14.0] * 3
        assert list(f["arinc_poly/value"][:]) == [2.0, 5.0] * 3
        idx = f["measurement_index"][:]
        assert len(idx) == 5 and (idx["samples"] > 0).all()
    print("CSV definitions (1553, ARINC-429): OK")

    # Blank channel: every channel, combined; duplicate copies removed; repeated names combined
    src3 = os.path.join(tmp, "bus.ch10")
    with open(src3, "wb") as fh:
        fh.write(bus_file())
    defs3 = os.path.join(tmp, "defs3.csv")
    with open(defs3, "w") as fh:
        fh.write("name,type,channel,rt,tr,sa,word,rate,units\n"
                 "Counter,1553,,9,1,2,1,50,n\n"
                 "Twice,1553,2,9,1,2,1,,n\n"
                 "Twice,1553,BUS-B,9,1,2,1,,n\n")
    out3 = os.path.join(tmp, "bus.h5")
    logs = []
    convert(src3, out3, year=2026, definitions=defs3, log=logs.append, include_raw=False)
    with h5py.File(out3, "r") as f:
        assert "raw" not in f                                # raw dump left out
        c = f["Counter"]
        assert list(c["raw"][:]) == list(range(200))         # 0-99 once (not twice) + 100-199
        assert c.attrs["duplicates_removed"] == 100
        assert list(c.attrs["sources"]) == ["ch0002_RT9_T_SA2_W1", "ch0003_RT9_T_SA2_W1",
                                            "ch0004_RT9_T_SA2_W1"]
        assert len(c["ch0003_RT9_T_SA2_W1/raw"]) == 100
        assert abs(c.attrs["rate_hz_observed"] - 10) < 0.01
        assert "50 Hz" in c.attrs["rate_warning"]
        t2 = f["Twice"]                                      # channel 2 + channel named BUS-B (= 4)
        assert list(t2["raw"][:]) == list(range(200)) and t2.attrs["duplicates_removed"] == 0
        assert len(t2.attrs["sources"]) == 2
    assert not [p for p in os.listdir(tmp) if "partial" in p]  # scratch file cleaned up
    print("blank channel, duplicates, repeated names, rate check, no raw dump: OK")

    data, n = pcm_file()
    src2 = os.path.join(tmp, "pcm.ch10")
    with open(src2, "wb") as fh:
        fh.write(data)
    out2 = os.path.join(tmp, "pcm.h5")
    convert(src2, out2, year=2026)
    with h5py.File(out2, "r") as f:
        temp = f["TEMP"]
        assert temp.attrs["units"] == "degC"
        assert temp["ch0003_W3"].attrs["defined_in"] == "tmats"
        assert np.allclose(temp["value"][:], -40 + 0.5 * (100 + n))
        press = f["PRESS"]
        raw = np.sort(np.concatenate([n * 10, n * 10 + 5]))
        assert np.allclose(np.sort(press["raw"][:]), raw)
        assert np.allclose(np.sort(press["value"][:]), raw * 0.1)
        assert np.all(np.diff(press["time_ns"][:]) >= 0)
        assert list(f["STATUS_HI/raw"][:]) == [0xA] * len(n)
        # word 3 starts 48 bits (32-bit sync + word 2) after the frame start, at 96 kbit/s
        frame_t = f["raw/channels/ch0003_PCMF1/messages"]["time_ns"][:]
        assert list(temp["time_ns"][:] - frame_t) == [int(48 * 1e9 / 96000)] * len(n)
    print("TMATS PCM definitions: OK")

    with open(defs, "a") as fh:                              # PCM row with a blank channel
        fh.write("temp_any_channel,pcm,,,,,3,,,,,,,,,,,,,\n")
    both = os.path.join(tmp, "both.h5")
    convert_many([src, src2], both, year=2026, definitions=defs)
    with h5py.File(both, "r") as f:
        assert "half_word2" in f["sample"] and "TEMP" in f["pcm"] and "raw" in f["pcm"]
        assert list(f["pcm/temp_any_channel/raw"][:]) == list(100 + n)
    print("combined file measurements: OK")

    check_commutation(tmp)


if __name__ == "__main__":
    main()
