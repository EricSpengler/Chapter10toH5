"""Per-data-type decoding of the channel specific data word (CSDW) and the
intra-packet messages of Chapter 10 packets.

Every decoder returns messages as ``(fields, data_offset, data_length)`` where
the offsets point into the packet body (CSDW at offset 0). The raw body of
every packet is always kept, so a decoder that does not understand a packet
never loses data: it just produces no message rows.
"""

import struct


class DecodeError(Exception):
    pass


def bits(value, shift, width):
    return (value >> shift) & ((1 << width) - 1)


IPTS = struct.Struct("<Q")
U16 = struct.Struct("<H")
U32 = struct.Struct("<I")


class Decoder:
    """Base decoder: raw body only, CSDW kept as a plain 32-bit value."""

    name = "unknown"
    description = "Unknown / reserved data type"
    # (column, shift, width) extracted from the CSDW
    csdw_fields = ()
    # (column, numpy dtype string) produced per message, in order
    msg_fields = ()
    has_csdw = True

    def csdw_values(self, csdw):
        return tuple(bits(csdw, s, w) for _n, s, w in self.csdw_fields)

    def messages(self, body, csdw, pkt):
        return []


def _finish(off, body_len, count, expected):
    """Validate that a message walk consumed the body exactly."""
    if off > body_len or body_len - off > 3:
        raise DecodeError("walk ended at %d of %d bytes" % (off, body_len))
    if expected is not None and count != expected:
        raise DecodeError("found %d messages, CSDW says %d" % (count, expected))


def _walk_with_pads(walker, body, csdw, pads):
    """Try each padding rule and keep the one that consumes the body exactly."""
    last = None
    for pad in pads:
        try:
            return walker(body, csdw, pad)
        except (DecodeError, struct.error) as exc:
            last = exc
    raise DecodeError(str(last))


def _pad(n, pad):
    return (n + pad - 1) // pad * pad if pad > 1 else n


# ---------------------------------------------------------------- computer

class ComputerF0(Decoder):
    name = "ComputerF0_UserDefined"
    description = "Computer generated F0, user defined"


class TMATS(Decoder):
    name = "ComputerF1_TMATS"
    description = "Computer generated F1, setup record (TMATS)"
    csdw_fields = (("ch10_version", 0, 8), ("config_change", 8, 1), ("xml_format", 9, 1))
    msg_fields = ()


class RecordingEvents(Decoder):
    name = "ComputerF2_Events"
    description = "Computer generated F2, recording events"
    csdw_fields = (("event_count", 0, 12), ("ipdh_present", 31, 1))
    msg_fields = (("ipts", "u8"), ("ipdh_time", "u8"), ("event_number", "u2"),
                  ("event_count", "u2"), ("event_occurrence", "u1"))

    def messages(self, body, csdw, pkt):
        n = bits(csdw, 0, 12)
        ipdh = bits(csdw, 31, 1)
        off, out = 4, []
        for _ in range(n):
            ipts = IPTS.unpack_from(body, off)[0]
            off += 8
            t = 0
            if ipdh:
                t = IPTS.unpack_from(body, off)[0]
                off += 8
            e = U32.unpack_from(body, off)[0]
            off += 4
            out.append(((ipts, t, bits(e, 0, 12), bits(e, 12, 16), bits(e, 28, 1)), off - 4, 4))
        _finish(off, len(body), len(out), n)
        return out


class RecordingIndex(Decoder):
    name = "ComputerF3_Index"
    description = "Computer generated F3, recording index"
    csdw_fields = (("entry_count", 0, 16), ("ipdh_present", 29, 1),
                   ("file_size_present", 30, 1), ("index_type_node", 31, 1))
    # entry_kind: 0 root entry, 1 node entry, 2 previous-root link, 3 file size
    msg_fields = (("entry_kind", "u1"), ("ipts", "u8"), ("ipdh_time", "u8"),
                  ("channel_id", "u2"), ("data_type", "u1"), ("offset", "u8"))

    def messages(self, body, csdw, pkt):
        n = bits(csdw, 0, 16)
        ipdh, fsize, node = bits(csdw, 29, 1), bits(csdw, 30, 1), bits(csdw, 31, 1)
        off, out = 4, []
        if fsize:
            out.append(((3, 0, 0, 0, 0, IPTS.unpack_from(body, off)[0]), off, 8))
            off += 8
        for i in range(n):
            start = off
            ipts = IPTS.unpack_from(body, off)[0]
            off += 8
            t = 0
            if ipdh:
                t = IPTS.unpack_from(body, off)[0]
                off += 8
            cid = dt = 0
            if node:
                e = U32.unpack_from(body, off)[0]
                off += 4
                cid, dt = bits(e, 0, 16), bits(e, 16, 8)
            o = IPTS.unpack_from(body, off)[0]
            off += 8
            # the last root index entry points back to the previous root index packet
            kind = 1 if node else (2 if i == n - 1 else 0)
            out.append(((kind, ipts, t, cid, dt, o), start, off - start))
        _finish(off, len(body), len(out), None)
        return out


class ComputerF4(Decoder):
    name = "ComputerF4_StreamingConfig"
    description = "Computer generated F4, streaming configuration"


# ---------------------------------------------------------------- PCM

class PCMF1(Decoder):
    name = "PCMF1"
    description = "PCM F1"
    csdw_fields = (("sync_offset", 0, 18), ("unpacked", 18, 1), ("packed", 19, 1),
                   ("throughput", 20, 1), ("alignment_32bit", 21, 1), ("major_frame_lock", 24, 2),
                   ("minor_frame_lock", 26, 2), ("minor_frame_indicator", 28, 1),
                   ("major_frame_indicator", 29, 1), ("iph", 30, 1))
    msg_fields = (("ipts", "u8"), ("lock_status", "u1"))

    def __init__(self):
        self._frame_cache = {}

    @staticmethod
    def _iph_word(body, off, iph):
        return (U32 if iph == 12 else U16).unpack_from(body, off + 8)[0]

    @staticmethod
    def _fits(body, iph, flen):
        """Frames of flen tile the body with rising times and clean IPH words."""
        stride = iph + flen
        prev = -1
        for off in range(4, len(body), stride):
            t = IPTS.unpack_from(body, off)[0]
            if t <= prev or PCMF1._iph_word(body, off, iph) & ~0xF000F000:
                return False
            prev = t
        return True

    def messages(self, body, csdw, pkt):
        """Split into minor frames when intra-packet headers are present.

        Frame length is not stored in the packet, so it is found by trying
        every candidate length and keeping the smallest one that tiles the
        body exactly with monotonically increasing time stamps.
        """
        if not bits(csdw, 30, 1) or bits(csdw, 20, 1):
            return []
        align = 4 if bits(csdw, 21, 1) else 2
        iph = 8 + align
        data = len(body) - 4
        key = (pkt.channel_id, len(body))
        best = self._frame_cache.get(key)
        if best is None or not self._fits(body, iph, best):
            best = None
            for flen in range(align, data - iph + 1, align):
                if data % (iph + flen) == 0 and self._fits(body, iph, flen):
                    best = flen
                    break
            self._frame_cache[key] = best
        if best is None:
            raise DecodeError("could not determine minor frame length")
        out, off = [], 4
        while off < len(body):
            ipts = IPTS.unpack_from(body, off)[0]
            w = self._iph_word(body, off, iph)
            lock = ((w >> 12) | (w >> 28)) & 0xF
            out.append(((ipts, lock), off + iph, best))
            off += iph + best
        return out


# ---------------------------------------------------------------- time

class TimeF1(Decoder):
    name = "TimeF1"
    description = "Time F1 (IRIG / GPS / RTC)"
    csdw_fields = (("time_source", 0, 4), ("time_format", 4, 4), ("leap_year", 8, 1),
                   ("date_format_dmy", 9, 1), ("irig_time_source", 12, 4))
    msg_fields = (("year", "u2"), ("month", "u1"), ("day", "u1"), ("day_of_year", "u2"),
                  ("hour", "u1"), ("minute", "u1"), ("second", "u1"), ("millisecond", "u2"))

    def messages(self, body, csdw, pkt):
        dmy = bits(csdw, 9, 1)
        w = struct.unpack_from("<%dH" % (4 if dmy else 3), body, 4)
        ms = bits(w[0], 4, 4) * 100 + bits(w[0], 0, 4) * 10
        sec = bits(w[0], 12, 3) * 10 + bits(w[0], 8, 4)
        minute = bits(w[1], 4, 3) * 10 + bits(w[1], 0, 4)
        hour = bits(w[1], 12, 2) * 10 + bits(w[1], 8, 4)
        if dmy:
            day = bits(w[2], 4, 4) * 10 + bits(w[2], 0, 4)
            month = bits(w[2], 12, 1) * 10 + bits(w[2], 8, 4)
            year = (bits(w[3], 12, 2) * 1000 + bits(w[3], 8, 4) * 100
                    + bits(w[3], 4, 4) * 10 + bits(w[3], 0, 4))
            doy = 0
        else:
            doy = bits(w[2], 8, 2) * 100 + bits(w[2], 4, 4) * 10 + bits(w[2], 0, 4)
            day = month = year = 0
        return [((year, month, day, doy, hour, minute, sec, ms), 4, len(body) - 4)]


class TimeF2(Decoder):
    name = "TimeF2_Network"
    description = "Time F2, network time (NTP / PTP)"
    csdw_fields = (("time_status", 0, 4), ("time_format", 4, 4))
    msg_fields = (("seconds", "u4"), ("subseconds", "u4"))

    def messages(self, body, csdw, pkt):
        s, sub = struct.unpack_from("<II", body, 4)
        return [((s, sub), 4, len(body) - 4)]


# ---------------------------------------------------------------- 1553

class MS1553F1(Decoder):
    name = "MIL-STD-1553F1"
    description = "MIL-STD-1553 F1"
    csdw_fields = (("message_count", 0, 24), ("time_tag_bits", 30, 2))
    msg_fields = (("ipts", "u8"), ("block_status", "u2"), ("bus_b", "u1"), ("message_error", "u1"),
                  ("rt_to_rt", "u1"), ("format_error", "u1"), ("response_timeout", "u1"),
                  ("word_count_error", "u1"), ("sync_type_error", "u1"), ("invalid_word_error", "u1"),
                  ("gap1", "u1"), ("gap2", "u1"), ("command_word", "u2"), ("rt", "u1"),
                  ("transmit", "u1"), ("subaddress", "u1"), ("word_count_or_mode", "u1"),
                  ("command_word2", "u2"))
    HDR = struct.Struct("<QHHH")

    def messages(self, body, csdw, pkt):
        n = bits(csdw, 0, 24)
        hdr = self.HDR
        off, out = 4, []
        blen = len(body)
        for _ in range(n):
            ipts, bsw, gap, ln = hdr.unpack_from(body, off)
            off += 14
            if off + ln > blen:
                raise DecodeError("1553 message overruns packet")
            cw = U16.unpack_from(body, off)[0] if ln >= 2 else 0
            rt2rt = (bsw >> 11) & 1
            cw2 = U16.unpack_from(body, off + 2)[0] if rt2rt and ln >= 4 else 0
            out.append(((ipts, bsw, (bsw >> 13) & 1, (bsw >> 12) & 1, rt2rt, (bsw >> 10) & 1,
                         (bsw >> 9) & 1, (bsw >> 5) & 1, (bsw >> 4) & 1, (bsw >> 3) & 1,
                         gap & 0xFF, gap >> 8, cw, cw >> 11, (cw >> 10) & 1, (cw >> 5) & 0x1F,
                         cw & 0x1F, cw2), off, ln))
            off += ln + (ln & 1)
        _finish(off, blen, len(out), n)
        return out


class MS1553F2(Decoder):
    name = "MIL-STD-1553F2_16PP194"
    description = "MIL-STD-1553 F2 (16PP194)"
    csdw_fields = (("message_count", 0, 32),)
    msg_fields = (("ipts", "u8"), ("status", "u2"))

    def messages(self, body, csdw, pkt):
        n = csdw
        off, out = 4, []
        for _ in range(n):
            ipts, ln, st = struct.unpack_from("<QHH", body, off)
            off += 12
            out.append(((ipts, st), off, ln))
            off += ln
        _finish(off, len(body), len(out), n)
        return out


# ---------------------------------------------------------------- analog / discrete

class AnalogF1(Decoder):
    name = "AnalogF1"
    description = "Analog F1"
    csdw_fields = (("packing_mode", 0, 2), ("sample_bits", 2, 6), ("subchannel", 8, 8),
                   ("total_subchannels", 16, 8), ("sample_factor", 24, 4), ("same", 28, 1))


class DiscreteF1(Decoder):
    name = "DiscreteF1"
    description = "Discrete F1"
    csdw_fields = (("mode", 0, 3), ("length", 3, 5))
    msg_fields = (("ipts", "u8"), ("states", "u4"))

    def messages(self, body, csdw, pkt):
        off, out = 4, []
        while off + 12 <= len(body):
            ipts, st = struct.unpack_from("<QI", body, off)
            out.append(((ipts, st), off + 8, 4))
            off += 12
        _finish(off, len(body), len(out), None)
        return out


# ---------------------------------------------------------------- generic message

class MessageF0(Decoder):
    name = "MessageF0"
    description = "Message data F0"
    csdw_fields = (("message_count", 0, 16), ("packet_type", 16, 2))
    msg_fields = (("ipts", "u8"), ("subchannel", "u2"), ("format_error", "u1"), ("data_error", "u1"))

    def messages(self, body, csdw, pkt):
        n = bits(csdw, 0, 16)
        off, out = 4, []
        for _ in range(n):
            ipts, iph = struct.unpack_from("<QI", body, off)
            off += 12
            ln = iph & 0xFFFF
            out.append(((ipts, bits(iph, 16, 14), bits(iph, 30, 1), bits(iph, 31, 1)), off, ln))
            off += _pad(ln, 2)
        _finish(off, len(body), len(out), n)
        return out


# ---------------------------------------------------------------- ARINC 429

# ARINC-429 labels are sent MSB first, so the conventional octal label is the
# bit-reversed low byte of the recorded word.
LABEL_OCTAL = [int(oct(int("{:08b}".format(i)[::-1], 2))[2:]) for i in range(256)]


class ARINC429F0(Decoder):
    name = "ARINC429F0"
    description = "ARINC-429 F0"
    csdw_fields = (("message_count", 0, 16),)
    PAIR = struct.Struct("<II")
    msg_fields = (("gap_time", "u4"), ("bus_speed_high", "u1"), ("parity_error", "u1"),
                  ("format_error", "u1"), ("bus", "u1"), ("word", "u4"), ("label", "u1"),
                  ("label_octal", "u2"), ("sdi", "u1"), ("data", "u4"), ("ssm", "u1"), ("parity", "u1"))

    def messages(self, body, csdw, pkt):
        n = bits(csdw, 0, 16)
        off, out = 4, []
        pair = self.PAIR
        octal = LABEL_OCTAL
        for _ in range(n):
            iph, word = pair.unpack_from(body, off)
            label = word & 0xFF
            out.append(((iph & 0xFFFFF, (iph >> 21) & 1, (iph >> 22) & 1, (iph >> 23) & 1,
                         iph >> 24, word, label, octal[label], (word >> 8) & 3,
                         (word >> 10) & 0x7FFFF, (word >> 29) & 3, word >> 31), off + 4, 4))
            off += 8
        _finish(off, len(body), len(out), n)
        return out


# ---------------------------------------------------------------- video / image

class _TS188(Decoder):
    """MPEG-2 transport stream packets, optionally preceded by an IPTS."""
    msg_fields = (("ipts", "u8"),)
    iph_bit = None

    def messages(self, body, csdw, pkt):
        iph = bits(csdw, self.iph_bit, 1)
        stride = 188 + (8 if iph else 0)
        data = len(body) - 4
        if data % stride:
            raise DecodeError("body is not a whole number of TS packets")
        out, off = [], 4
        while off < len(body):
            ipts = None
            if iph:
                ipts = IPTS.unpack_from(body, off)[0]
                off += 8
            out.append(((ipts,), off, 188))
            off += 188
        return out


class VideoF0(_TS188):
    name = "VideoF0_MPEG2TS"
    description = "Video F0, MPEG-2 transport stream"
    csdw_fields = (("byte_alignment", 23, 1), ("payload_type", 24, 4), ("klv", 28, 1),
                   ("scr_rtc_sync", 29, 1), ("iph", 30, 1), ("embedded_time", 31, 1))
    iph_bit = 30


class VideoF1(_TS188):
    name = "VideoF1_ISO13818"
    description = "Video F1, ISO 13818-1 MPEG-2"
    csdw_fields = (("packet_count", 0, 12), ("transport_type", 12, 1), ("mode", 13, 1),
                   ("embedded_time", 14, 1), ("encoding_profile", 15, 4), ("iph", 19, 1),
                   ("scr_rtc_sync", 20, 1), ("klv", 21, 1))
    iph_bit = 19


class VideoF2(VideoF1):
    name = "VideoF2_ISO14496"
    description = "Video F2, ISO 14496 MPEG-4 part 10"
    csdw_fields = VideoF1.csdw_fields + (("encoding_level", 22, 4), ("audio_encoding", 26, 1))


class VideoRaw(Decoder):
    name = "Video"
    description = "Video (other format)"


class ImageRaw(Decoder):
    name = "Image"
    description = "Image"


# ---------------------------------------------------------------- UART

class UARTF0(Decoder):
    name = "UARTF0"
    description = "UART F0"
    csdw_fields = (("ipts_present", 31, 1),)
    msg_fields = (("ipts", "u8"), ("subchannel", "u2"), ("parity_error", "u1"))

    def messages(self, body, csdw, pkt):
        has_ts = bits(csdw, 31, 1)
        off, out = 4, []
        blen = len(body)
        while off + 4 <= blen:
            ipts = None
            if has_ts:
                ipts = IPTS.unpack_from(body, off)[0]
                off += 8
            iph = U32.unpack_from(body, off)[0]
            off += 4
            ln = iph & 0xFFFF
            if off + ln > blen:
                raise DecodeError("UART message overruns packet")
            out.append(((ipts, bits(iph, 16, 14), bits(iph, 31, 1)), off, ln))
            off += _pad(ln, 2)
        _finish(off, blen, len(out), None)
        return out


# ---------------------------------------------------------------- IEEE 1394 / parallel

class I1394F0(Decoder):
    name = "IEEE1394F0"
    description = "IEEE-1394 F0 (transaction)"
    csdw_fields = (("transaction_count", 0, 16), ("sync_code", 16, 4), ("packet_body_type", 29, 3))


class I1394F1(Decoder):
    name = "IEEE1394F1"
    description = "IEEE-1394 F1 (physical layer)"
    csdw_fields = (("message_count", 0, 16),)
    msg_fields = (("ipts", "u8"), ("status", "u1"), ("speed", "u1"), ("overflow_error", "u1"),
                  ("buffer_overflow", "u1"))

    def _walk(self, body, csdw, pad):
        n = bits(csdw, 0, 16)
        off, out = 4, []
        for _ in range(n):
            ipts, iph = struct.unpack_from("<QI", body, off)
            off += 12
            ln = iph & 0xFFFF
            out.append(((ipts, bits(iph, 24, 8), bits(iph, 16, 4), bits(iph, 20, 2),
                         bits(iph, 22, 1)), off, ln))
            off += _pad(ln, pad)
        _finish(off, len(body), len(out), n)
        return out

    def messages(self, body, csdw, pkt):
        return _walk_with_pads(self._walk, body, csdw, (4, 2))


class ParallelF0(Decoder):
    name = "ParallelF0"
    description = "Parallel F0"
    csdw_fields = (("scan_number", 0, 24), ("type", 24, 8))


# ---------------------------------------------------------------- Ethernet

class EthernetF0(Decoder):
    name = "EthernetF0"
    description = "Ethernet F0"
    csdw_fields = (("frame_count", 0, 16), ("time_tag_bits", 25, 3), ("format", 28, 4))
    msg_fields = (("ipts", "u8"), ("length_error", "u1"), ("data_crc_error", "u1"),
                  ("network_id", "u1"), ("speed", "u1"), ("content", "u1"),
                  ("frame_error", "u1"), ("frame_crc_error", "u1"),
                  ("dst_mac", "u8"), ("src_mac", "u8"), ("ethertype", "u2"))

    def _walk(self, body, csdw, pad):
        n = bits(csdw, 0, 16)
        off, out = 4, []
        blen = len(body)
        for _ in range(n):
            ipts, iph = struct.unpack_from("<QI", body, off)
            off += 12
            ln = iph & 0x3FFF
            if off + ln > blen:
                raise DecodeError("Ethernet frame overruns packet")
            dst = src = et = 0
            if ln >= 14 and bits(iph, 28, 2) == 0:
                dst = int.from_bytes(body[off:off + 6], "big")
                src = int.from_bytes(body[off + 6:off + 12], "big")
                et = int.from_bytes(body[off + 12:off + 14], "big")
            out.append(((ipts, bits(iph, 14, 1), bits(iph, 15, 1), bits(iph, 16, 8),
                         bits(iph, 24, 4), bits(iph, 28, 2), bits(iph, 30, 1), bits(iph, 31, 1),
                         dst, src, et), off, ln))
            off += _pad(ln, pad)
        _finish(off, blen, len(out), n)
        return out

    def messages(self, body, csdw, pkt):
        return _walk_with_pads(self._walk, body, csdw, (2, 4, 1))


class EthernetF1(Decoder):
    name = "EthernetF1_ARINC664"
    description = "Ethernet F1, ARINC-664 / UDP payload"
    csdw_fields = (("message_count", 0, 16), ("iph_length", 16, 16))
    msg_fields = (("ipts", "u8"), ("flags", "u1"), ("error", "u1"), ("virtual_link", "u2"),
                  ("src_ip", "u4"), ("dst_ip", "u4"), ("dst_port", "u2"), ("src_port", "u2"))

    def _walk(self, body, csdw, pad):
        n = bits(csdw, 0, 16)
        iphl = bits(csdw, 16, 16) or 28
        off, out = 4, []
        for _ in range(n):
            ipts, flags, err, ln, vl, _r, sip, dip, dport, sport = struct.unpack_from(
                "<QBBHHHIIHH", body, off)
            off += iphl
            out.append(((ipts, flags, err, vl, sip, dip, dport, sport), off, ln))
            off += _pad(ln, pad)
        _finish(off, len(body), len(out), n)
        return out

    def messages(self, body, csdw, pkt):
        return _walk_with_pads(self._walk, body, csdw, (4, 2, 1))


# ---------------------------------------------------------------- CAN

class CANF0(Decoder):
    name = "CANBusF0"
    description = "CAN bus F0"
    csdw_fields = (("message_count", 0, 16),)
    msg_fields = (("ipts", "u8"), ("subchannel", "u2"), ("format_error", "u1"),
                  ("data_error", "u1"), ("can_id", "u4"), ("rtr", "u1"), ("extended_id", "u1"))

    def _walk(self, body, csdw, pad):
        n = bits(csdw, 0, 16)
        off, out = 4, []
        for _ in range(n):
            ipts, iph, idw = struct.unpack_from("<QII", body, off)
            ln = iph & 0xF  # bytes, including the 4-byte ID word
            off += 12
            out.append(((ipts, bits(iph, 16, 14), bits(iph, 30, 1), bits(iph, 31, 1),
                         bits(idw, 0, 29), bits(idw, 30, 1), bits(idw, 31, 1)),
                        off + 4, max(ln - 4, 0)))
            off += _pad(max(ln, 4), pad)
        _finish(off, len(body), len(out), n)
        return out

    def messages(self, body, csdw, pkt):
        return _walk_with_pads(self._walk, body, csdw, (2, 4))


# ---------------------------------------------------------------- registry

def _raw(name, description):
    return type(name, (Decoder,), {"name": name, "description": description})


DECODERS = {
    0x00: ComputerF0(), 0x01: TMATS(), 0x02: RecordingEvents(), 0x03: RecordingIndex(),
    0x04: ComputerF4(),
    0x09: PCMF1(),
    0x11: TimeF1(), 0x12: TimeF2(),
    0x19: MS1553F1(), 0x1A: MS1553F2(),
    0x21: AnalogF1(), 0x29: DiscreteF1(),
    0x30: MessageF0(), 0x38: ARINC429F0(),
    0x40: VideoF0(), 0x41: VideoF1(), 0x42: VideoF2(),
    0x43: _raw("VideoF3", "Video F3")(), 0x44: _raw("VideoF4", "Video F4")(),
    0x48: _raw("ImageF0", "Image F0")(), 0x49: _raw("ImageF1", "Still imagery F1")(),
    0x4A: _raw("ImageF2", "Dynamic imagery F2")(),
    0x50: UARTF0(),
    0x58: I1394F0(), 0x59: I1394F1(),
    0x60: ParallelF0(),
    0x68: EthernetF0(), 0x69: EthernetF1(),
    0x70: _raw("TSPI_CTS_F0", "TSPI/CTS F0 (GPS NMEA-RTCM)")(),
    0x71: _raw("TSPI_CTS_F1", "TSPI/CTS F1 (EAG ACMI)")(),
    0x72: _raw("TSPI_CTS_F2", "TSPI/CTS F2 (ACTTS)")(),
    0x78: CANF0(),
    0x79: _raw("FibreChannelF0", "Fibre Channel F0")(),
    0x7A: _raw("FibreChannelF1", "Fibre Channel F1")(),
}


def decoder_for(data_type):
    d = DECODERS.get(data_type)
    if d is None:
        d = _raw("Type0x%02X" % data_type, "Unknown / reserved data type 0x%02X" % data_type)()
    return d
