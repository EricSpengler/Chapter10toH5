"""Write a synthetic Chapter 10 file that exercises every supported data type.

Usage: python tests/make_sample.py [output.ch10]

The file includes TMATS, time, 1553, PCM, ARINC-429, UART, Ethernet F0/F1,
message, discrete, analog, video, CAN, 1394, recording events and index
packets, an unknown data type, secondary headers, data checksums, a burst of
junk bytes in the middle and a truncated packet at the end.
"""

import struct
import sys

RTC_HZ = 10_000_000
EXPECT = {}  # filled in for the tests


def header_sum(b):
    return sum(struct.unpack("<%dH" % (len(b) // 2), b)) & 0xFFFF


class Writer:
    def __init__(self):
        self.out = bytearray()
        self.seq = {}

    def packet(self, cid, dtype, body, rtc, secondary=None, sec_fmt=0, ipts_src=0, cks=0, ver=0x07):
        seq = self.seq.get(cid, 0)
        self.seq[cid] = (seq + 1) & 0xFF
        csize = (0, 1, 2, 4)[cks]
        filler = b"\x00" * ((-(len(body) + csize)) % 4)
        payload = body + filler
        if cks == 1:
            payload += bytes([sum(payload) & 0xFF])
        elif cks == 2:
            payload += struct.pack("<H", sum(struct.unpack("<%dH" % (len(payload) // 2), payload)) & 0xFFFF)
        elif cks == 3:
            payload += struct.pack("<I", sum(struct.unpack("<%dI" % (len(payload) // 4), payload)) & 0xFFFFFFFF)
        flags = (0x80 if secondary is not None else 0) | (ipts_src << 6) | (sec_fmt << 2) | cks
        hlen = 36 if secondary is not None else 24
        hdr = struct.pack("<HHIIBBBB", 0xEB25, cid, hlen + len(payload), len(body), ver, seq, flags, dtype)
        hdr += rtc.to_bytes(6, "little")
        hdr += struct.pack("<H", header_sum(hdr))
        if secondary is not None:
            sec = struct.pack("<QH", secondary, 0)
            hdr += sec + struct.pack("<H", header_sum(sec))
        self.out += hdr + payload
        return len(self.out) - len(hdr) - len(payload)


def bcd_time_doy(doy, h, m, s, ms):
    w0 = ((ms // 10) % 10) | ((ms // 100) << 4) | ((s % 10) << 8) | ((s // 10) << 12)
    w1 = (m % 10) | ((m // 10) << 4) | ((h % 10) << 8) | ((h // 10) << 12)
    w2 = (doy % 10) | (((doy // 10) % 10) << 4) | ((doy // 100) << 8)
    return struct.pack("<HHH", w0, w1, w2)


def build():
    w = Writer()
    rtc0 = 0x0000_1234_0000
    tmats = ("G\\PN:Synthetic;\nG\\106:17;\nR-1\\ID:REC;\nR-1\\N:3;\n"
             "R-1\\TK1-1:2;\nR-1\\CDT-1:1553IN;\nR-1\\DSI-1:BUS1;\n").encode()
    w.packet(0, 0x01, struct.pack("<I", 0x07) + tmats, rtc0)
    EXPECT["tmats"] = tmats.decode()

    # Time F1, day-of-year format, day 100 12:00:00.000, one per second
    EXPECT["time"] = []
    for k in range(3):
        rtc = rtc0 + k * RTC_HZ
        w.packet(1, 0x11, struct.pack("<I", 0x0000_0011) + bcd_time_doy(100, 12, 0, k, 0), rtc)
        EXPECT["time"].append((rtc, (100, 12, 0, k, 0)))

        # 1553: one BC->RT receive (cmd + 4 words + status) and one RT-RT
        msgs = b""
        cw = (5 << 11) | (0 << 10) | (3 << 5) | 4
        data = struct.pack("<6H", cw, 1, 2, 3, 4, (5 << 11))
        msgs += struct.pack("<QHHH", rtc + 100, 0, 0x0403, len(data)) + data
        cw_rx = (6 << 11) | (0 << 10) | (1 << 5) | 2
        cw_tx = (7 << 11) | (1 << 10) | (1 << 5) | 2
        data2 = struct.pack("<6H", cw_rx, cw_tx, 7 << 11, 0xAAAA, 0x5555, 6 << 11)
        msgs += struct.pack("<QHHH", rtc + 200, (1 << 13) | (1 << 11), 0x0000, len(data2)) + data2
        w.packet(2, 0x19, struct.pack("<I", 2) + msgs, rtc + 300, cks=2)

        # PCM F1, IPH present, 16-bit alignment, packed, 5 minor frames of 20 bytes
        csdw = (1 << 30) | (1 << 19) | (3 << 26) | (3 << 24)
        frames = b""
        for f in range(5):
            frames += struct.pack("<QH", rtc + 1000 + f * 50, 0xF000) + bytes([0xFE, 0x6B, 0x28, 0x40]) + bytes(range(f, f + 16))
        w.packet(3, 0x09, struct.pack("<I", csdw) + frames, rtc + 1500)

        # ARINC-429: 3 words on bus 2
        words = b""
        for i in range(3):
            iph = (100 * i) | (1 << 21) | (2 << 24)
            word = 0o203 if i == 0 else (0x31 | (i << 10))
            words += struct.pack("<II", iph, word)
        w.packet(4, 0x38, struct.pack("<I", 3) + words, rtc + 2000)

        # UART with IPTS in IEEE-1588 format via secondary header
        sec_time = ((1_700_000_000 + k) << 32) | 500
        payload = b""
        for i, text in enumerate([b"hello", b"world!"]):
            ipts = ((1_700_000_000 + k) << 32) | (1000 * (i + 1))
            payload += struct.pack("<QI", ipts, len(text) | (3 << 16)) + text + (b"\x00" if len(text) % 2 else b"")
        w.packet(5, 0x50, struct.pack("<I", 1 << 31) + payload, rtc + 2500,
                 secondary=sec_time, sec_fmt=1, ipts_src=1)

    # Ethernet F0: two frames, one odd length
    frames = b""
    for i, ln in enumerate((60, 61)):
        frame = bytes.fromhex("ffffffffffff") + bytes.fromhex("0200000000%02x" % i) + b"\x08\x00" + bytes(ln - 14)
        iph = ln | (1 << 16)
        frames += struct.pack("<QI", rtc0 + 5000 + i, iph) + frame + (b"\x00" if ln % 2 else b"")
    w.packet(6, 0x68, struct.pack("<I", 2) + frames, rtc0 + 5100)

    # Junk bytes between packets (simulated corruption)
    junk_off = len(w.out)
    w.out += b"\xde\xad\xbe\xef" * 5 + b"\x25\xeb\x00"
    EXPECT["junk"] = (junk_off, 23)

    # Message F0
    msg = b"abc"
    w.packet(7, 0x30, struct.pack("<I", 1) + struct.pack("<QI", rtc0 + 6000, len(msg) | (9 << 16)) + msg + b"\x00",
             rtc0 + 6000)
    # Discrete F1: two samples
    w.packet(8, 0x29, struct.pack("<I", 0) + struct.pack("<QIQI", rtc0 + 7000, 0x5, rtc0 + 7100, 0xA), rtc0 + 7200)
    # Analog F1 (raw only)
    w.packet(9, 0x21, struct.pack("<I", (16 << 2) | (1 << 16)) + struct.pack("<8h", *range(8)), rtc0 + 8000)
    # Video F0 with IPH: two TS packets
    ts = b""
    for i in range(2):
        ts += struct.pack("<Q", rtc0 + 9000 + i) + b"\x47" + bytes(187)
    w.packet(10, 0x40, struct.pack("<I", 1 << 30) + ts, rtc0 + 9100)
    # CAN: one extended frame with 8 data bytes
    can = struct.pack("<QII", rtc0 + 10000, 12 | (4 << 16), 0x1ABCDEF | (1 << 31)) + bytes(range(8))
    w.packet(11, 0x78, struct.pack("<I", 1) + can, rtc0 + 10000)
    # Unknown / reserved type
    w.packet(12, 0x7F, struct.pack("<I", 0xCAFEF00D) + b"opaque", rtc0 + 11000, cks=1)
    # Ethernet F1 (ARINC-664)
    udp = b"payload!!"
    iph = struct.pack("<QBBHHHIIHH", rtc0 + 12000, 0, 0, len(udp), 42, 0, 0x0A000001, 0x0A000002, 5000, 6000)
    w.packet(13, 0x69, struct.pack("<I", 1 | (28 << 16)) + iph + udp + b"\x00" * 3, rtc0 + 12000)
    # IEEE-1394 F1
    w.packet(14, 0x59, struct.pack("<I", 1) + struct.pack("<QI", rtc0 + 13000, 8 | (2 << 16)) + bytes(8),
             rtc0 + 13000, cks=3)
    # Recording event and root index
    w.packet(0, 0x02, struct.pack("<I", 1) + struct.pack("<QI", rtc0 + 14000, 7 | (3 << 12) | (1 << 28)), rtc0 + 14000)
    w.packet(0, 0x03, struct.pack("<I", 2) + struct.pack("<QQQQ", rtc0 + 15000, 0x100, rtc0 + 15000, 0),
             rtc0 + 15000)
    # Deliberately broken 1553 packet: CSDW claims 5 messages, body holds none
    w.packet(2, 0x19, struct.pack("<I", 5), rtc0 + 16000)

    # Truncated packet at the end of file (recording cut off)
    tail_off = len(w.out)
    w.packet(15, 0x19, struct.pack("<I", 0) + bytes(40), rtc0 + 17000)
    del w.out[tail_off + 30:]
    EXPECT["tail"] = (tail_off, 30)
    return bytes(w.out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.ch10"
    data = build()
    with open(path, "wb") as f:
        f.write(data)
    print("wrote %s (%d bytes)" % (path, len(data)))
