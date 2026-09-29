"""Convert a Chapter 10 file into an HDF5 file holding every packet.

Layout of the output file::

    /                         attrs: source file, sizes, counts, converter version
    /file_index               one row per packet, in file order
    /summary                  one row per (channel, data type) group
    /TMATS/text_NNN           each TMATS setup record as text
    /TMATS/attributes         key/value table parsed from the first TMATS record
    /unparsed/regions         byte ranges that could not be framed as packets
    /unparsed/bytes           those bytes, verbatim
    /channels/chCCCC_<Type>/
        packets               decoded packet headers + CSDW fields, one row per packet
        body                  every packet's data body (CSDW + data) back to back, uint8
        messages              decoded intra-packet messages (where the type has them)

``packets.body_offset`` / ``body_length`` and ``messages.data_offset`` /
``data_length`` index into ``body``, so raw bytes are stored exactly once.
"""

import bisect
import datetime as dt
import os
import time

import h5py
import numpy as np

from . import __version__
from .decoders import decoder_for
from .measurements import load_csv, write_measurements
from .packets import Ch10Reader
from .tmats import parse_tmats

NO_TIME = -1
# ipts_format value for messages without their own time stamp (time_ns is the packet time)
NO_IPTS = 255


class Cancelled(Exception):
    """Raised when the caller sets the cancel event."""


def _int_dtype(width):
    return "u1" if width <= 8 else "u2" if width <= 16 else "u4"


PACKET_FIELDS = [
    ("file_packet_index", "u8"), ("file_offset", "u8"), ("packet_length", "u4"),
    ("data_length", "u4"), ("header_version", "u1"), ("sequence", "u1"), ("flags", "u1"),
    ("secondary_header", "u1"), ("ipts_source", "u1"), ("rtc_sync_error", "u1"),
    ("data_overflow", "u1"), ("secondary_time_format", "u1"), ("checksum_type", "u1"),
    ("rtc", "u8"), ("secondary_time", "u8"), ("header_checksum_ok", "u1"),
    ("secondary_checksum_ok", "i1"), ("data_checksum_ok", "i1"), ("time_ns", "i8"),
    ("csdw", "u4"), ("body_offset", "u8"), ("body_length", "u4"),
    ("message_count", "u4"), ("decode_status", "u1"),
]
# decode_status: 0 = messages decoded, 1 = no message decoder for this type
# (raw body only), 2 = decoder failed on this packet (raw body only)

FILE_INDEX_DTYPE = np.dtype([
    ("file_offset", "u8"), ("channel_id", "u2"), ("data_type", "u1"),
    ("packet_length", "u4"), ("data_length", "u4"), ("rtc", "u8"), ("time_ns", "i8"),
    ("group_packet_index", "u8"),
])


class Table:
    """Buffered, resizable 1-D compound dataset."""

    def __init__(self, parent, name, dtype, chunk_rows, comp):
        self.dtype = np.dtype(dtype)
        self.ds = parent.create_dataset(name, shape=(0,), maxshape=(None,), dtype=self.dtype,
                                        chunks=(chunk_rows,), **comp)
        self.rows = []
        self.n = 0
        self.limit = chunk_rows * 4

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.limit:
            self.flush()

    def __len__(self):
        return self.n + len(self.rows)

    def flush(self):
        if not self.rows:
            return
        arr = np.array(self.rows, dtype=self.dtype)
        self.ds.resize((self.n + len(arr),))
        self.ds[self.n:] = arr
        self.n += len(arr)
        self.rows = []


class Blob:
    """Buffered, resizable uint8 dataset."""

    def __init__(self, parent, name, comp, chunk=1 << 20):
        self.ds = parent.create_dataset(name, shape=(0,), maxshape=(None,), dtype="u1",
                                        chunks=(chunk,), **comp)
        self.buf = bytearray()
        self.n = 0
        self.limit = chunk * 8

    @property
    def size(self):
        return self.n + len(self.buf)

    def append(self, data):
        off = self.size
        self.buf += data
        if len(self.buf) >= self.limit:
            self.flush()
        return off

    def flush(self):
        if not self.buf:
            return
        arr = np.frombuffer(bytes(self.buf), dtype="u1")
        self.ds.resize((self.n + len(arr),))
        self.ds[self.n:] = arr
        self.n += len(arr)
        self.buf = bytearray()


class Group:
    """Output for one (channel id, data type) pair."""

    def __init__(self, root, channel_id, data_type, comp):
        self.decoder = dec = decoder_for(data_type)
        self.name = "ch%04d_%s" % (channel_id, dec.name)
        g = self.h5 = root.create_group(self.name)
        g.attrs["channel_id"] = channel_id
        g.attrs["data_type"] = data_type
        g.attrs["data_type_hex"] = "0x%02X" % data_type
        g.attrs["data_type_name"] = dec.name
        g.attrs["description"] = dec.description
        csdw = [(n, _int_dtype(w)) for n, _s, w in dec.csdw_fields]
        self.packets = Table(g, "packets", PACKET_FIELDS + [("csdw_" + n, t) for n, t in csdw], 4096, comp)
        self.body = Blob(g, "body", comp)
        self.messages = None
        self.has_ipts = False
        if dec.msg_fields:
            fields = [("packet_index", "u8"), ("file_packet_index", "u8")] + list(dec.msg_fields)
            self.has_ipts = fields[2][0] == "ipts"
            if self.has_ipts:
                fields.append(("ipts_format", "u1"))
            fields += [("time_ns", "i8"), ("data_offset", "u8"), ("data_length", "u4")]
            self.messages = Table(g, "messages", fields, 16384, comp)
        self.decode_errors = 0
        self.first_error = ""

    def flush(self):
        self.packets.flush()
        self.body.flush()
        if self.messages is not None:
            self.messages.flush()

    def finish(self):
        self.flush()
        g = self.h5
        g.attrs["packet_count"] = len(self.packets)
        g.attrs["message_count"] = len(self.messages) if self.messages is not None else 0
        g.attrs["decode_errors"] = self.decode_errors
        if self.first_error:
            g.attrs["first_decode_error"] = self.first_error
        if self.messages is None:
            g.attrs["note"] = ("No intra-packet message decoder for this type; every packet's "
                               "raw data is in 'body'.")


# ---------------------------------------------------------------- time

def time_f1_to_ns(fields, csdw, year):
    """Absolute time (ns since Unix epoch) from a decoded Time F1 message."""
    yr, month, day, doy, hour, minute, sec, ms = fields
    try:
        if (csdw >> 9) & 1:
            base = dt.datetime(yr, month, day, tzinfo=dt.timezone.utc)
        else:
            base = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1)
    except ValueError:
        return NO_TIME
    t = base + dt.timedelta(hours=hour, minutes=minute, seconds=sec, milliseconds=ms)
    return int(t.timestamp()) * 1_000_000_000 + t.microsecond * 1000


class TimeBase:
    """Maps 48-bit RTC values to absolute time using the file's time packets.

    Each packet uses the nearest time packet at or before it in the file (or
    the first one, for packets before any time packet).
    """

    def __init__(self, refs):
        refs = sorted(r for r in refs if r[2] != NO_TIME)
        self.idx = [r[0] for r in refs]
        self.rtc = [r[1] for r in refs]
        self.ns = [r[2] for r in refs]

    def __bool__(self):
        return bool(self.idx)

    def rtc_to_ns(self, file_index, rtc):
        if not self.idx:
            return NO_TIME
        i = max(bisect.bisect_right(self.idx, file_index) - 1, 0)
        d = (rtc - self.rtc[i]) & 0xFFFFFFFFFFFF
        if d >= 1 << 47:
            d -= 1 << 48
        return self.ns[i] + d * 100

    def ipts_to_ns(self, file_index, ipts, fmt):
        if fmt == 0:                        # 48-bit RTC
            return self.rtc_to_ns(file_index, ipts & 0xFFFFFFFFFFFF)
        if fmt == 2:                        # IEEE-1588: seconds high, nanoseconds low
            return (ipts >> 32) * 1_000_000_000 + (ipts & 0xFFFFFFFF)
        return NO_TIME                      # Ch4 binary / ERTC: left raw


def prescan_time(path, year):
    """Collect (file_packet_index, rtc, time_ns) for every Time F1 packet."""
    dec = decoder_for(0x11)
    found = []
    with Ch10Reader(path, verify_data_checksum=False) as rd:
        for p in rd:
            if p.data_type != 0x11:
                continue
            body = rd.body(p)
            try:
                csdw = int.from_bytes(body[:4], "little")
                fields = dec.messages(body, csdw, p)[0][0]
            except Exception:
                continue
            finally:
                body.release()
            found.append((p.index, p.rtc, csdw, fields))
    # year_known is a description of where the year came from, or None if guessed
    year_known = "entered by user" if year is not None else None
    if year is None:
        dmy = [f[0] for _i, _r, c, f in found if (c >> 9) & 1 and f[0]]
        if dmy:
            year, year_known = dmy[0], "from time packets"
    if year is None:
        year = dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc).year
    refs = [(i, r, time_f1_to_ns(f, c, year)) for i, r, c, f in found]
    return refs, year, year_known


# ---------------------------------------------------------------- conversion

def convert(in_path, out_path, year=None, compress=True, progress=None, cancel=None, log=None,
            definitions=None, include_raw=True):
    """Convert one Chapter 10 file to HDF5; its content sits at the file root."""
    return convert_many([in_path], out_path, year, compress, progress, cancel, log, definitions,
                        include_raw)[0]


def default_output(in_paths, outdir=None):
    """Default .h5 path: <name>.h5 for one input, <first name>_combined.h5 for several."""
    first = in_paths[0]
    stem = os.path.splitext(os.path.basename(first))[0]
    name = stem + (".h5" if len(in_paths) == 1 else "_combined.h5")
    return os.path.join(outdir or os.path.dirname(os.path.abspath(first)), name)


def source_group_names(in_paths):
    """Unique HDF5 group names for each input file, based on the file name."""
    names, seen = [], {"sources"}
    for path in in_paths:
        base = os.path.splitext(os.path.basename(path))[0].replace("/", "_") or "file"
        name, k = base, 2
        while name in seen:
            name, k = "%s_%d" % (base, k), k + 1
        seen.add(name)
        names.append(name)
    return names


def convert_many(in_paths, out_path, year=None, compress=True, progress=None, cancel=None, log=None,
                 definitions=None, include_raw=True):
    """Convert Chapter 10 files into a single HDF5 file. Returns one summary dict per input.

    Measurements (from the definitions CSV and the TMATS) are top-level groups,
    one per measurement name. The full dump of the recording goes under /raw,
    or is left out when include_raw is False.

    With one input, this layout is at the file root. With several, each input
    gets its own top-level group (named after the file) holding the same
    layout, and the root lists them in /sources.

    progress(done_bytes, total_bytes) covers all inputs; cancel is an optional
    threading.Event; log(str) receives human-readable messages. definitions is
    an optional CSV of measurement definitions (see measurements.py).
    """
    log = log or (lambda msg: None)
    defs = []
    if definitions:
        defs, problems = load_csv(definitions)
        for msg in problems:
            log("Definitions file, " + msg)
        log("Loaded %d measurement definitions from %s." % (len(defs), os.path.basename(definitions)))
    comp = {"compression": "gzip", "compression_opts": 1, "shuffle": True} if compress else {}
    sizes = [os.path.getsize(p) for p in in_paths]
    total = sum(sizes) or 1
    names = source_group_names(in_paths)
    tmp_path = out_path + ".partial"
    results = []
    try:
        with h5py.File(tmp_path, "w") as h5:
            done = 0
            for path, size, name in zip(in_paths, sizes, names):
                target = h5 if len(in_paths) == 1 else h5.create_group(name)
                sub = (lambda d, t, base=done: progress(base + d, total)) if progress else None
                res = _convert_one(path, target, size, year, comp, sub, cancel, log, defs,
                                   include_raw, tmp_path + ".raw%d" % len(results))
                res["group"] = target.name
                results.append(res)
                done += size
            if len(in_paths) > 1:
                str_t = h5py.string_dtype("utf-8")
                h5.create_dataset("sources", data=np.array(
                    [(n, os.path.basename(p), s, r["packets"]) for n, p, s, r in zip(names, in_paths, sizes, results)],
                    dtype=[("group", str_t), ("source_file", str_t), ("source_size", "u8"), ("packets", "u8")]))
                h5.attrs["source_files"] = [os.path.basename(p) for p in in_paths]
                h5.attrs["file_count"] = len(in_paths)
                h5.attrs["converter"] = "ch10toh5 %s" % __version__
                h5.attrs["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    os.replace(tmp_path, out_path)
    for r in results:
        r["output"] = out_path
    if progress:
        progress(total, total)
    return results


def _convert_one(in_path, h5, size, year, comp, progress, cancel, log, defs=(), include_raw=True,
                 scratch_path=None):
    t0 = time.time()
    log("Scanning time packets in %s..." % os.path.basename(in_path))
    refs, year, year_known = prescan_time(in_path, year)
    tb = TimeBase(refs)
    if tb:
        log("Found %d time packets%s." % (len(refs), "" if year_known else
                                          " (day-of-year format, assuming year %d)" % year))
    else:
        log("No usable time packets: time_ns columns will be -1.")
    groups = {}
    # The raw dump is always built (measurements are read from it); when it is not
    # wanted in the output it goes to a scratch file that is deleted afterwards.
    scratch = None if include_raw else h5py.File(scratch_path, "w")
    try:
        raw = h5.create_group("raw") if include_raw else scratch
        n_packets, gap_rows = _write(in_path, raw, size, year, year_known, tb,
                                     comp if include_raw else {}, groups, progress, cancel, log)
        for k, v in raw.attrs.items():
            h5.attrs[k] = v
        if include_raw:
            h5.attrs["raw_dump"] = "/raw holds every packet of the recording"
        pairs = [(k.decode(), v.decode()) for k, v in raw["TMATS/attributes"][:]] if "TMATS/attributes" in raw else []
        n_meas = write_measurements(raw, h5, defs, pairs, comp, log)
    finally:
        if scratch is not None:
            scratch.close()
            os.remove(scratch_path)
    if not include_raw and not n_meas:
        log("Warning: the raw dump was left out and no measurements are defined, so this file is nearly empty.")
    elapsed = time.time() - t0
    log("Done: %d packets in %d channel groups, %.1f s." % (n_packets, len(groups), elapsed))
    return {
        "measurements": n_meas,
        "packets": n_packets,
        "groups": len(groups),
        "unparsed_bytes": sum(n for _o, n, _b in gap_rows),
        "decode_errors": sum(g.decode_errors for g in groups.values()),
        "seconds": elapsed,
    }


def _write(in_path, h5, size, year, year_known, tb, comp, groups, progress, cancel, log):
    gaps = []
    tmats_texts = []
    last_report = 0.0
    n_packets = 0
    with Ch10Reader(in_path, on_gap=lambda o, n: gaps.append((o, n))) as rd:
        chan_root = h5.create_group("channels")
        index = Table(h5, "file_index", FILE_INDEX_DTYPE, 16384, comp)
        unparsed = h5.create_group("unparsed")
        gap_bytes = Blob(unparsed, "bytes", comp, chunk=1 << 16)
        gap_rows = []
        buf = rd.buf

        log("Converting %s (%.1f MB)..." % (os.path.basename(in_path), size / 1e6))
        for p in rd:
            while gaps:
                o, n = gaps.pop(0)
                gap_rows.append((o, n, gap_bytes.append(buf[o:o + n])))
                log("Unframed bytes at offset %d (%d bytes) kept in %s." % (o, n, unparsed.name))

            key = (p.channel_id, p.data_type)
            g = groups.get(key)
            if g is None:
                g = groups[key] = Group(chan_root, p.channel_id, p.data_type, comp)
            dec = g.decoder
            body = rd.body(p)
            csdw = int.from_bytes(body[:4], "little") if len(body) >= 4 else 0
            body_off = g.body.append(body)
            pkt_index = len(g.packets)
            ptime = tb.rtc_to_ns(p.index, p.rtc)

            status, msgs = 1, []
            if g.messages is not None:
                try:
                    msgs = dec.messages(body, csdw, p)
                    status = 0
                except Exception as exc:  # any decoder failure keeps the raw body
                    status, msgs = 2, []
                    g.decode_errors += 1
                    if not g.first_error:
                        g.first_error = "packet %d: %s" % (pkt_index, exc)
                        log("%s: could not decode messages (%s); raw data kept." % (g.name, exc))

            if msgs:
                fmt = 0 if not p.ipts_source else 1 + p.secondary_time_format
                if p.data_type == 0x11:
                    t_abs = time_f1_to_ns(msgs[0][0], csdw, year)
                gap_sum = 0
                for fields, doff, dlen in msgs:
                    if g.has_ipts:
                        if fields[0] is None:   # this packet carries no intra-packet time stamps
                            fields = (0,) + fields[1:]
                            g.messages.append((pkt_index, p.index) + fields + (NO_IPTS, ptime, body_off + doff, dlen))
                        else:
                            t = tb.ipts_to_ns(p.index, fields[0], fmt)
                            g.messages.append((pkt_index, p.index) + fields + (fmt, t, body_off + doff, dlen))
                    else:
                        if p.data_type == 0x11:
                            t = t_abs
                        elif p.data_type == 0x38 and ptime != NO_TIME:
                            # ARINC-429: packet RTC is the first word; gap_time (0.1 us) is
                            # measured from the start of the preceding word
                            gap_sum += fields[0]
                            t = ptime + gap_sum * 100
                        else:
                            t = ptime
                        g.messages.append((pkt_index, p.index) + fields + (t, body_off + doff, dlen))

            if p.data_type == 0x01:
                tmats_texts.append((p.index, bytes(body[4:]).decode("latin-1").rstrip("\x00")))

            g.packets.append((
                p.index, p.offset, p.packet_length, p.data_length, p.header_version, p.sequence,
                p.flags, p.has_secondary, p.ipts_source, p.rtc_sync_error, p.data_overflow,
                p.secondary_time_format, p.checksum_type, p.rtc, p.secondary_time,
                p.header_checksum_ok, p.secondary_checksum_ok, p.data_checksum_ok, ptime,
                csdw, body_off, p.data_length, len(msgs), status,
            ) + dec.csdw_values(csdw))
            index.append((p.offset, p.channel_id, p.data_type, p.packet_length, p.data_length,
                          p.rtc, ptime, pkt_index))
            body.release()
            n_packets += 1

            if n_packets % 2000 == 0:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                now = time.time()
                if progress and now - last_report > 0.1:
                    progress(p.offset + p.packet_length, size)
                    last_report = now

        for o, n in gaps:
            gap_rows.append((o, n, gap_bytes.append(buf[o:o + n])))
            log("Unframed bytes at offset %d (%d bytes) kept in %s." % (o, n, unparsed.name))
        gap_bytes.flush()
        unparsed.create_dataset("regions", data=np.array(
            gap_rows, dtype=[("file_offset", "u8"), ("length", "u8"), ("bytes_offset", "u8")]))
        unparsed.attrs["note"] = "Byte ranges that did not form valid Chapter 10 packets."

        index.flush()
        for g in groups.values():
            g.finish()

        # TMATS
        tm = h5.create_group("TMATS")
        str_t = h5py.string_dtype("utf-8")
        for i, (fpi, text) in enumerate(tmats_texts):
            d = tm.create_dataset("text_%03d" % i, data=text, dtype=str_t)
            d.attrs["file_packet_index"] = fpi
        if tmats_texts:
            pairs = parse_tmats(tmats_texts[0][1])
            tm.create_dataset("attributes", data=np.array(pairs, dtype=[("key", str_t), ("value", str_t)])
                              if pairs else np.zeros(0, dtype=[("key", str_t), ("value", str_t)]))

        # Summary table
        summary = []
        for (cid, dtp), g in sorted(groups.items()):
            summary.append((cid, dtp, g.decoder.name, g.h5.name, len(g.packets),
                            len(g.messages) if g.messages is not None else 0, g.body.size,
                            g.decode_errors))
        h5.create_dataset("summary", data=np.array(summary, dtype=[
            ("channel_id", "u2"), ("data_type", "u1"), ("type_name", str_t), ("path", str_t),
            ("packets", "u8"), ("messages", "u8"), ("body_bytes", "u8"), ("decode_errors", "u8")]))

        a = h5.attrs
        a["source_file"] = os.path.basename(in_path)
        a["source_size"] = size
        a["packet_count"] = n_packets
        a["unparsed_bytes"] = sum(n for _o, n, _b in gap_rows)
        a["converter"] = "ch10toh5 %s" % __version__
        a["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        a["time_year"] = year
        a["time_year_source"] = year_known or "assumed (file modification year)"
        a["time_ns_note"] = ("time_ns = nanoseconds since 1970-01-01 UTC derived from Time F1 "
                             "packets and the 10 MHz RTC; -1 where unknown.")
    return n_packets, gap_rows
