"""IRIG 106 Chapter 10 packet framing: header parsing, checksums and resync.

All multi-byte fields in Chapter 10 are little-endian. Bit numbers in the
comments below follow the standard: bit 0 is the least significant bit of the
little-endian word.
"""

import mmap
import struct

SYNC = 0xEB25
SYNC_BYTES = b"\x25\xeb"
HEADER = struct.Struct("<HHIIBBBB6sH")      # 24-byte primary header
SECONDARY = struct.Struct("<QHH")           # 12-byte secondary header
MAX_PACKET = 512 * 1024 * 1024              # sanity cap for a single packet

CHECKSUM_SIZES = (0, 1, 2, 4)
SECONDARY_TIME_FORMATS = {0: "IRIG106 Ch4 binary", 1: "IEEE-1588", 2: "ERTC", 3: "reserved"}


def header_checksum(buf, off, nbytes):
    """16-bit sum of the little-endian words covering nbytes at off."""
    return sum(struct.unpack_from("<%dH" % (nbytes // 2), buf, off)) & 0xFFFF


class Packet:
    """One Chapter 10 packet located inside the mapped file."""

    __slots__ = (
        "index", "offset", "channel_id", "packet_length", "data_length",
        "header_version", "sequence", "flags", "data_type", "rtc",
        "header_checksum", "header_checksum_ok", "secondary_time",
        "secondary_checksum_ok", "body_start", "data_checksum_ok",
    )

    @property
    def has_secondary(self):
        return (self.flags >> 7) & 1

    @property
    def ipts_source(self):
        """0 = intra-packet times are RTC, 1 = secondary header time format."""
        return (self.flags >> 6) & 1

    @property
    def rtc_sync_error(self):
        return (self.flags >> 5) & 1

    @property
    def data_overflow(self):
        return (self.flags >> 4) & 1

    @property
    def secondary_time_format(self):
        return (self.flags >> 2) & 3

    @property
    def checksum_type(self):
        return self.flags & 3


class Ch10Reader:
    """Iterates over packets of a Chapter 10 file.

    Bytes that cannot be framed as packets (corruption, truncated tail) are
    reported through ``on_gap(offset, length)`` so nothing in the file is lost.
    """

    def __init__(self, path, on_gap=None, verify_data_checksum=True):
        self.path = path
        self._f = open(path, "rb")
        self.size = self._f.seek(0, 2)
        self.buf = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ) if self.size else b""
        self.on_gap = on_gap or (lambda off, ln: None)
        self.verify_data_checksum = verify_data_checksum
        self.pos = 0

    def close(self):
        if isinstance(self.buf, mmap.mmap):
            self.buf.close()
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _plausible(self, off):
        """Return the parsed header tuple if a believable packet starts at off."""
        buf, size = self.buf, self.size
        if off + 24 > size:
            return None
        h = HEADER.unpack_from(buf, off)
        sync, _cid, plen, dlen, _ver, _seq, flags = h[:7]
        if sync != SYNC or plen < 28 or plen > MAX_PACKET or off + plen > size:
            return None
        hdr_len = 36 if flags & 0x80 else 24
        if dlen + hdr_len > plen:
            return None
        return h

    def _accept(self, off, h):
        """Decide whether the header at off is a real packet."""
        if header_checksum(self.buf, off, 22) == h[9]:
            return True
        # Bad header checksum: trust it only if the next packet lines up.
        nxt = off + h[2]
        return nxt == self.size or self._plausible(nxt) is not None

    def __iter__(self):
        buf, size = self.buf, self.size
        index = 0
        pos = 0
        while pos < size:
            h = self._plausible(pos)
            if h is None or not self._accept(pos, h):
                # Resync: scan forward for the next believable header.
                start = pos
                pos = buf.find(SYNC_BYTES, pos + 1)
                while pos != -1:
                    h = self._plausible(pos)
                    if h is not None and self._accept(pos, h):
                        break
                    pos = buf.find(SYNC_BYTES, pos + 1)
                if pos == -1:
                    self.on_gap(start, size - start)
                    return
                self.on_gap(start, pos - start)

            p = Packet()
            p.index = index
            p.offset = pos
            (_sync, p.channel_id, p.packet_length, p.data_length, p.header_version,
             p.sequence, p.flags, p.data_type, rtc, p.header_checksum) = h
            p.rtc = int.from_bytes(rtc, "little")
            p.header_checksum_ok = int(header_checksum(buf, pos, 22) == p.header_checksum)
            body = pos + 24
            if p.flags & 0x80:
                p.secondary_time, _res, sec_sum = SECONDARY.unpack_from(buf, body)
                p.secondary_checksum_ok = int(header_checksum(buf, body, 10) == sec_sum)
                body += 12
            else:
                p.secondary_time = 0
                p.secondary_checksum_ok = -1
            p.body_start = body
            p.data_checksum_ok = self._check_data(p) if self.verify_data_checksum else -1
            yield p
            index += 1
            pos += p.packet_length
            self.pos = pos

    def _check_data(self, p):
        ctype = p.flags & 3
        if ctype == 0:
            return -1
        n = CHECKSUM_SIZES[ctype]
        end = p.offset + p.packet_length - n
        region = memoryview(self.buf)[p.body_start:end]
        try:
            if ctype == 1:
                total = sum(region) & 0xFF
                stored = self.buf[end]
            elif ctype == 2:
                total = sum(region[: len(region) // 2 * 2].cast("H")) & 0xFFFF
                stored = struct.unpack_from("<H", self.buf, end)[0]
            else:
                total = sum(region[: len(region) // 4 * 4].cast("I")) & 0xFFFFFFFF
                stored = struct.unpack_from("<I", self.buf, end)[0]
        finally:
            region.release()
        return int(total == stored)

    def body(self, p):
        """The packet's data body (CSDW + data), data_length bytes long."""
        return memoryview(self.buf)[p.body_start:p.body_start + p.data_length]
