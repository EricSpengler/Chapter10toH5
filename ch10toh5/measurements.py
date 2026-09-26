"""Engineering-unit measurements: turn raw 1553 words, ARINC-429 words and PCM
frame words into named time/value arrays.

Definitions come from two places:

* a CSV file (usually built from the ICD), one row per measurement:
  ``name,type,channel,rt,tr,sa,word,words,label,sdi,bus,word_interval,lsb,bits,
  encoding,scale,offset,coefficients,units,description``
  (only ``name`` and ``type`` plus the selector columns for that type are required)
* the recording's own TMATS, for PCM: D-group word locations and C-group
  conversions, where the TMATS carries them.

Each measurement is written to ``measurements/<name>/`` as ``time_ns``,
``value`` (engineering units, float64) and ``raw`` (the extracted bit field).
"""

import csv
import re
from dataclasses import dataclass, field

import h5py
import numpy as np

NO_TIME = -1
ENCODINGS = ("unsigned", "signed", "signmag", "offset")


@dataclass
class Definition:
    name: str
    kind: str                       # "1553", "arinc429" or "pcm"
    channel: int = None
    rt: int = None
    tr: int = None
    sa: int = None
    word: int = None                # 1553 data word (1 = first data word); PCM word position (1 = sync)
    words: int = 1                  # 1553: combine this many consecutive words, most significant first
    label: int = None               # ARINC-429 label, written in octal digits (e.g. 203)
    sdi: int = None
    bus: int = None
    word_interval: int = 0          # PCM: repeat every N words within the minor frame
    lsb: int = None                 # bit field: lowest bit (0 = least significant)
    bits: int = None                # bit field: width
    encoding: str = "unsigned"
    scale: float = 1.0
    offset: float = 0.0
    coefficients: list = field(default_factory=list)   # polynomial a0 + a1*x + a2*x^2 ...
    pairs: list = field(default_factory=list)          # (raw, eu) table, linear interpolation
    units: str = ""
    description: str = ""
    source: str = "csv"

    def selector(self):
        if self.kind == "1553":
            s = "RT %s %s SA %s word %s" % (self.rt, {0: "R", 1: "T"}.get(self.tr, "R/T"), self.sa, self.word)
            if self.words > 1:
                s += "-%d" % (self.word + self.words - 1)
        elif self.kind == "arinc429":
            s = "label %03d" % self.label + ("" if self.sdi is None else " SDI %d" % self.sdi)
            if self.bus is not None:
                s += " bus %d" % self.bus
        else:
            s = "word %s" % self.word + (" every %d" % self.word_interval if self.word_interval else "")
        if self.channel is not None:
            s = "channel %d %s" % (self.channel, s)
        return s


# ---------------------------------------------------------------- CSV

def _int(v, octal=False):
    v = (v or "").strip()
    if not v:
        return None
    if octal:
        return int(v.lstrip("0o") or "0")    # kept as octal digits to match label_octal
    return int(v, 0)


def _float(v, default):
    v = (v or "").strip()
    return float(v) if v else default


def load_csv(path):
    """Read measurement definitions. Returns (definitions, problems)."""
    defs, problems = [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        lines = [(i, line) for i, line in enumerate(fh, start=1)
                 if line.strip() and not line.lstrip().startswith("#")]
        rows = csv.DictReader(line for _i, line in lines)
        for k, row in enumerate(rows, start=1):
            n = lines[k][0] if k < len(lines) else "?"
            row = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
            try:
                kind = row.get("type", "").strip().lower().replace("-", "").replace("_", "")
                kind = {"1553": "1553", "milstd1553": "1553", "arinc429": "arinc429", "arinc": "arinc429",
                        "429": "arinc429", "pcm": "pcm"}.get(kind)
                if not row.get("name", "").strip() or kind is None:
                    raise ValueError("needs a name and a type of 1553, arinc429 or pcm")
                d = Definition(
                    name=row["name"].strip(), kind=kind, channel=_int(row.get("channel")),
                    rt=_int(row.get("rt")), tr=_int(row.get("tr")), sa=_int(row.get("sa")),
                    word=_int(row.get("word")), words=_int(row.get("words")) or 1,
                    label=_int(row.get("label"), octal=True), sdi=_int(row.get("sdi")), bus=_int(row.get("bus")),
                    word_interval=_int(row.get("word_interval")) or 0,
                    lsb=_int(row.get("lsb")), bits=_int(row.get("bits")),
                    encoding=(row.get("encoding", "").strip().lower() or "unsigned"),
                    scale=_float(row.get("scale"), 1.0), offset=_float(row.get("offset"), 0.0),
                    coefficients=[float(c) for c in re.split(r"[;\s]+", row.get("coefficients", "").strip()) if c],
                    units=row.get("units", "").strip(), description=row.get("description", "").strip())
                if d.encoding not in ENCODINGS:
                    raise ValueError("encoding must be one of " + ", ".join(ENCODINGS))
                if kind == "1553" and None in (d.rt, d.sa, d.word):
                    raise ValueError("1553 rows need rt, sa and word")
                if kind == "arinc429" and d.label is None:
                    raise ValueError("arinc429 rows need a label")
                if kind == "pcm" and None in (d.channel, d.word):
                    raise ValueError("pcm rows need channel and word")
                defs.append(d)
            except ValueError as exc:
                problems.append("line %d: %s" % (n, exc))
    return defs, problems


# ---------------------------------------------------------------- TMATS

def _tm_dict(pairs):
    return {k.upper(): v for k, v in pairs}


def pcm_formats(pairs):
    """Map Chapter 10 channel id -> PCM frame format, from TMATS R, M and P groups."""
    tm = _tm_dict(pairs)
    p_by_name = {}
    for k, v in tm.items():
        m = re.match(r"^P-(\d+)\\DLN$", k)
        if m:
            p_by_name[v.strip()] = m.group(1)
    m_links = {}      # M-x\ID data source id -> baseband data link name
    for k, v in tm.items():
        m = re.match(r"^M-(\d+)\\ID$", k)
        if m and ("M-%s\\BB\\DLN" % m.group(1)) in tm:
            m_links[v.strip()] = tm["M-%s\\BB\\DLN" % m.group(1)].strip()
    out = {}
    for k, v in tm.items():
        m = re.match(r"^R-(\d+)\\TK1-(\d+)$", k)
        if not m:
            continue
        dsi = tm.get("R-%s\\DSI-%s" % m.groups(), "").strip()
        p = p_by_name.get(dsi) or p_by_name.get(m_links.get(dsi, ""))
        if p is None:
            continue
        try:
            fmt = {
                "p_group": int(p), "link_name": tm["P-%s\\DLN" % p],
                "word_bits": int(tm["P-%s\\F1" % p]),
                "words": int(tm["P-%s\\MF1" % p]),
                "frame_bits": int(tm["P-%s\\MF2" % p]),
                "sync_bits": int(tm["P-%s\\MF4" % p]),
                "sync": tm.get("P-%s\\MF5" % p, "").strip(),
                "bit_rate": float(tm.get("P-%s\\D2" % p, "0") or 0),
            }
        except (KeyError, ValueError):
            continue
        out[int(v)] = fmt
    return out


def tmats_definitions(pairs, log=lambda m: None):
    """PCM measurement definitions from TMATS D (location) and C (conversion) groups."""
    tm = _tm_dict(pairs)
    formats = pcm_formats(pairs)
    chan_by_link = {f["link_name"].strip(): cid for cid, f in formats.items()}
    conv = {}
    for k, v in tm.items():
        m = re.match(r"^C-(\d+)\\DCN$", k)
        if m:
            conv[v.strip()] = m.group(1)
    defs = []
    for k, v in tm.items():
        m = re.match(r"^D-(\d+)\\MN-(\d+)-(\d+)$", k)
        if not m:
            continue
        x, y, n = m.groups()
        name = v.strip()
        link = tm.get("D-%s\\DLN" % x, "").strip()
        cid = chan_by_link.get(link)
        lt = tm.get("D-%s\\LT-%s-%s" % (x, y, n), "WDFR").strip().upper()
        wp = tm.get("D-%s\\WP-%s-%s-1" % (x, y, n))
        if cid is None or wp is None or lt != "WDFR":
            log("TMATS measurement %s skipped (location type %s or no PCM link)." % (name, lt))
            continue
        fp = int(tm.get("D-%s\\FP-%s-%s-1" % (x, y, n), "1") or 1)
        fi = int(tm.get("D-%s\\FI-%s-%s-1" % (x, y, n), "0") or 0)
        if fi > 1 or fp > 1:
            log("TMATS measurement %s skipped: subcommutated words are not decoded yet." % name)
            continue
        d = Definition(name=name, kind="pcm", channel=cid, word=int(wp),
                       word_interval=int(tm.get("D-%s\\WI-%s-%s-1" % (x, y, n), "0") or 0), source="tmats")
        mask = tm.get("D-%s\\WFM-%s-%s-1-1" % (x, y, n), "FW").strip().upper()
        if mask not in ("", "FW") and set(mask) <= {"0", "1"}:
            ones = [i for i, c in enumerate(reversed(mask)) if c == "1"]
            d.lsb, d.bits = min(ones), max(ones) - min(ones) + 1
        c = conv.get(name)
        if c is not None:
            d.units = tm.get("C-%s\\MN4" % c, "").strip()
            bfm = tm.get("C-%s\\BFM" % c, "UNS").strip().upper()
            d.encoding = {"TWO": "signed", "INT": "signed", "SIG": "signmag", "OFF": "offset"}.get(bfm, "unsigned")
            dct = tm.get("C-%s\\DCT" % c, "NON").strip().upper()
            if dct in ("COE", "NPC"):
                order = int(tm.get("C-%s\\CO\\N" % c, "0") or 0)
                d.coefficients = [float(tm.get("C-%s\\CO" % c, "0"))] + [
                    float(tm.get("C-%s\\CO-%d" % (c, i), "0")) for i in range(1, order + 1)]
            elif dct == "PRS" or dct == "PTS":
                npairs = int(tm.get("C-%s\\PS\\N" % c, "0") or 0)
                d.pairs = [(float(tm["C-%s\\PS1-%d" % (c, i)]), float(tm["C-%s\\PS2-%d" % (c, i)]))
                           for i in range(1, npairs + 1)
                           if "C-%s\\PS1-%d" % (c, i) in tm and "C-%s\\PS2-%d" % (c, i) in tm]
        defs.append(d)
    return defs


# ---------------------------------------------------------------- conversion

def _field(raw, d, default_lsb, default_bits):
    lsb = default_lsb if d.lsb is None else d.lsb
    bits = default_bits - lsb if d.bits is None else d.bits
    val = (raw.astype(np.uint64) >> np.uint64(lsb)) & np.uint64((1 << bits) - 1)
    val = val.astype(np.int64)
    if d.encoding == "signed":
        val = np.where(val >= 1 << (bits - 1), val - (1 << bits), val)
    elif d.encoding == "signmag":
        mag = val & ((1 << (bits - 1)) - 1)
        val = np.where(val >> (bits - 1), -mag, mag)
    elif d.encoding == "offset":
        val = val - (1 << (bits - 1))
    return val


def to_eu(val, d):
    x = val.astype(float)
    if d.pairs:
        raw, eu = zip(*sorted(d.pairs))
        return np.interp(x, raw, eu)
    if d.coefficients:
        return np.polynomial.polynomial.polyval(x, d.coefficients)
    return x * d.scale + d.offset


def _gather(body, starts, nbytes, block=64 << 20):
    """Read nbytes at each start offset from a uint8 dataset, in bounded blocks."""
    out = np.zeros((len(starts), nbytes), dtype=np.uint8)
    i = 0
    while i < len(starts):
        lo = int(starts[i])
        j = int(np.searchsorted(starts, lo + block, side="left"))
        j = max(j, i + 1)
        hi = int(starts[j - 1]) + nbytes
        buf = body[lo:hi]
        idx = (starts[i:j] - lo)[:, None] + np.arange(nbytes)
        out[i:j] = buf[idx]
        i = j
    return out


def _channels(root, data_type, channel):
    grp = root.get("channels")
    if grp is None:
        return
    for g in grp.values():
        if int(g.attrs["data_type"]) == data_type and (channel is None or int(g.attrs["channel_id"]) == channel):
            yield g


def extract_1553(root, d):
    times, raws = [], []
    for g in _channels(root, 0x19, d.channel):
        if "messages" not in g or not len(g["messages"]):
            continue
        m = g["messages"][:]
        cw2 = m["command_word2"].astype(np.int64)
        rt2 = m["rt_to_rt"] == 1
        sa_ok = (d.sa != 0) & (d.sa != 31)
        normal = (~rt2) & (m["rt"] == d.rt) & (m["subaddress"] == d.sa)
        if d.tr is not None:
            normal &= m["transmit"] == d.tr
        rx = rt2 & (m["rt"] == d.rt) & (m["subaddress"] == d.sa) & (d.tr in (None, 0))
        tx = rt2 & ((cw2 >> 11) == d.rt) & (((cw2 >> 5) & 31) == d.sa) & (d.tr in (None, 1))
        # index of data word 1 inside the message: after cmd (+status for transmit, +cmd2+status for RT-RT)
        first = np.where(rt2, 3, np.where(m["transmit"] == 1, 2, 1))
        idx = first + d.word - 1
        ok = (normal | rx | tx) & sa_ok & (m["message_error"] == 0) & \
             ((idx + d.words) * 2 <= m["data_length"])
        if not ok.any():
            continue
        starts = (m["data_offset"][ok] + 2 * idx[ok]).astype(np.int64)
        b = _gather(g["body"], starts, 2 * d.words)
        w = b.view("<u2").astype(np.uint64)
        raw = np.zeros(len(starts), dtype=np.uint64)
        for k in range(d.words):
            raw = (raw << np.uint64(16)) | w[:, k]
        times.append(m["time_ns"][ok])
        raws.append(raw)
    return _join(times, raws), 0, 16 * d.words


def extract_arinc(root, d):
    times, raws = [], []
    for g in _channels(root, 0x38, d.channel):
        if "messages" not in g or not len(g["messages"]):
            continue
        m = g["messages"].fields(["label_octal", "sdi", "bus", "word", "time_ns", "parity_error", "format_error"])[:]
        ok = (m["label_octal"] == d.label) & (m["format_error"] == 0)
        if d.sdi is not None:
            ok &= m["sdi"] == d.sdi
        if d.bus is not None:
            ok &= m["bus"] == d.bus
        times.append(m["time_ns"][ok])
        raws.append(m["word"][ok].astype(np.uint64))
    return _join(times, raws), 10, 29      # default field: data bits 11-29 (19 bits incl. sign)


def _join(times, raws):
    if not times:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint64)
    t, r = np.concatenate(times), np.concatenate(raws)
    order = np.argsort(t, kind="stable")
    return t[order], r[order]


# ---------------------------------------------------------------- PCM

def _bits_from_bytes(data, align32=False):
    """PCM bit stream from recorded bytes: 16-bit (or 32-bit) little-endian words, MSB first."""
    a = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else data
    n = 4 if align32 else 2
    a = a[: a.shape[-1] // n * n] if a.ndim == 1 else a[:, : a.shape[-1] // n * n]
    shape = a.shape
    a = a.reshape(shape[:-1] + (-1, n))[..., ::-1].reshape(shape)
    return np.unpackbits(a, axis=-1)


def _unpacked_to_bits(frames, fmt):
    """Unpacked mode: every word right-aligned in its own 16-bit container."""
    w = frames[:, : frames.shape[1] // 2 * 2].copy().view("<u2")
    sync_c = -(-fmt["sync_bits"] // 16)
    per = -(-fmt["word_bits"] // 16)
    parts = [np.unpackbits(w[:, :sync_c].astype(">u2").view(np.uint8), axis=1)[:, -fmt["sync_bits"]:]]
    for k in range(fmt["words"] - 1):
        c = w[:, sync_c + k * per: sync_c + (k + 1) * per].astype(">u2").view(np.uint8)
        parts.append(np.unpackbits(c, axis=1)[:, -fmt["word_bits"]:])
    return np.concatenate(parts, axis=1)


def _frames_iph(g, fmt, csdw):
    """Minor frames already split by the converter (intra-packet headers on)."""
    m = g["messages"].fields(["time_ns", "data_offset", "data_length"])[:]
    if not len(m):
        return
    flen = int(m["data_length"][0])
    ok = m["data_length"] == flen
    raw = _gather(g["body"], m["data_offset"][ok].astype(np.int64), flen)
    if csdw["unpacked"]:
        bits = _unpacked_to_bits(raw, fmt)
    else:
        bits = _bits_from_bytes(raw, csdw["alignment_32bit"])
    yield m["time_ns"][ok], bits[:, : fmt["frame_bits"]]


def _frames_throughput(g, fmt, csdw, max_bits=1 << 27):
    """Throughput mode: find minor frames in the raw bit stream by their sync pattern."""
    pk = g["packets"].fields(["time_ns", "body_offset", "body_length"])[:]
    sync = np.array([int(c) for c in fmt["sync"]], dtype=np.uint8)
    F, S = fmt["frame_bits"], len(sync)
    if not S or F <= S:
        return
    rate = fmt["bit_rate"]
    carry = np.zeros(0, dtype=np.uint8)
    carry_t = None                    # (time of carry[0], ns per bit)
    body = g["body"]
    i = 0
    while i < len(pk):
        # load a batch of packets
        chunks, anchors, total = [carry], [], len(carry)
        start_i = i
        while i < len(pk) and total < max_bits:
            o, n = int(pk["body_offset"][i]), int(pk["body_length"][i])
            b = _bits_from_bytes(body[o + 4:o + n], csdw["alignment_32bit"])
            anchors.append((total, int(pk["time_ns"][i])))
            chunks.append(b)
            total += len(b)
            i += 1
        bits = np.concatenate(chunks)
        # bit index -> time, from packet times (interpolated), else bit rate
        pos = np.array([a for a, _t in anchors], dtype=np.float64)
        tim = np.array([t for _a, t in anchors], dtype=np.float64)
        if carry_t is not None:
            pos = np.concatenate([[0.0], pos])
            tim = np.concatenate([[carry_t], tim])
        ns_per_bit = 1e9 / rate if rate else (
            np.median(np.diff(tim) / np.diff(pos)) if len(pos) > 1 else 0.0)

        def bit_time(ix):
            k = np.clip(np.searchsorted(pos, ix, side="right") - 1, 0, len(pos) - 1)
            return (tim[k] + (ix - pos[k]) * ns_per_bit).astype(np.int64)

        starts = []
        p = 0
        while p + F + S <= len(bits):
            win = np.lib.stride_tricks.sliding_window_view(bits[p:p + F + S], S)
            hits = np.nonzero((win == sync).all(axis=1))[0]
            if not len(hits):
                p += F
                continue
            p += int(hits[0])
            k = np.arange((len(bits) - p - S) // F + 1)
            cand = p + k * F
            good = (bits[cand[:, None] + np.arange(S)] == sync).all(axis=1)
            run = int(np.argmin(good)) if not good.all() else len(good)
            starts.extend(cand[:run].tolist())
            p = int(cand[run - 1]) + F if run else p + 1
        starts = [s for s in starts if s + F <= len(bits)]
        if starts:
            st = np.array(starts, dtype=np.int64)
            frames = bits[st[:, None] + np.arange(F)]
            yield bit_time(st), frames
            keep_from = int(st[-1]) + F
        else:
            keep_from = max(len(bits) - F - S, 0)
        carry = bits[keep_from:]
        carry_t = float(bit_time(np.array([keep_from]))[0]) if len(carry) else None
        if i == start_i:
            break


def extract_pcm(root, defs, formats, log):
    """All PCM definitions for one channel, sharing one pass over its frames."""
    results = {d.name: ([], []) for d in defs}
    cid = defs[0].channel
    fmt = formats.get(cid)
    if fmt is None:
        if any(True for _g in _channels(root, 0x09, cid)):
            log("PCM channel %s: no frame format in TMATS, so its measurements are skipped." % cid)
        return {d.name: (np.zeros(0, np.int64), np.zeros(0, np.uint64)) for d in defs}, fmt
    for g in _channels(root, 0x09, cid):
        pk = g["packets"]
        if not len(pk):
            continue
        csdw = {k: int(pk["csdw_" + k][0]) for k in ("throughput", "unpacked", "alignment_32bit")}
        has_frames = "messages" in g and len(g["messages"])
        frames = _frames_iph(g, fmt, csdw) if has_frames else _frames_throughput(g, fmt, csdw)
        ns_per_bit = 1e9 / fmt["bit_rate"] if fmt["bit_rate"] else 0.0
        for t, bits in frames:
            weights = 1 << np.arange(63, -1, -1, dtype=np.uint64)
            for d in defs:
                positions = [d.word]
                if d.word_interval:
                    positions = list(range(d.word, fmt["words"] + 1, d.word_interval))
                for wp in positions:
                    b0 = 0 if wp == 1 else fmt["sync_bits"] + (wp - 2) * fmt["word_bits"]
                    wl = fmt["sync_bits"] if wp == 1 else fmt["word_bits"]
                    if b0 + wl > bits.shape[1]:
                        continue
                    word = (bits[:, b0:b0 + wl].astype(np.uint64) * weights[64 - wl:]).sum(axis=1).astype(np.uint64)
                    results[d.name][0].append(t + int(b0 * ns_per_bit))
                    results[d.name][1].append(word)
    out = {}
    for d in defs:
        t, r = _join(*results[d.name])
        out[d.name] = (t, r)
    return out, fmt


# ---------------------------------------------------------------- writer

def _safe(name, used):
    base = re.sub(r"[/\\.]+", "_", name).strip() or "measurement"
    n, k = base, 2
    while n in used:
        n, k = "%s_%d" % (base, k), k + 1
    used.add(n)
    return n


def write_measurements(root, defs, tmats_pairs, comp, log):
    """Evaluate every definition against the converted group and write the results."""
    all_defs = list(defs)
    formats = pcm_formats(tmats_pairs) if tmats_pairs else {}
    if tmats_pairs:
        from_tm = tmats_definitions(tmats_pairs, log)
        if from_tm:
            log("TMATS defines %d PCM measurements." % len(from_tm))
        all_defs += from_tm
    if not all_defs:
        return 0
    out = root.create_group("measurements")
    out.attrs["note"] = ("Each measurement has time_ns (ns since 1970 UTC), value (engineering units) "
                         "and raw (the extracted bit field before conversion).")
    used, rows = set(), []

    pcm_by_channel = {}
    for d in all_defs:
        if d.kind == "pcm":
            pcm_by_channel.setdefault(d.channel, []).append(d)
    pcm_results = {}
    for cid, group in pcm_by_channel.items():
        res, _fmt = extract_pcm(root, group, formats, log)
        pcm_results.update({id(d): res[d.name] for d in group})

    for d in all_defs:
        if d.kind == "1553":
            (t, raw), lsb0, width = extract_1553(root, d)
        elif d.kind == "arinc429":
            (t, raw), lsb0, width = extract_arinc(root, d)
        else:
            (t, raw) = pcm_results[id(d)]
            fmt = formats.get(d.channel) or {}
            lsb0, width = 0, fmt.get("word_bits", 16) if d.word != 1 else fmt.get("sync_bits", 16)
        field_raw = _field(raw, d, lsb0, width)
        value = to_eu(field_raw, d)
        name = _safe(d.name, used)
        g = out.create_group(name)
        kw = comp if len(t) > 1024 else {}
        g.create_dataset("time_ns", data=t.astype(np.int64), **kw)
        g.create_dataset("value", data=value.astype(np.float64), **kw)
        g.create_dataset("raw", data=field_raw.astype(np.int64), **kw)
        a = g.attrs
        a["name"] = d.name
        a["units"] = d.units
        a["type"] = d.kind
        a["source"] = d.selector()
        a["defined_in"] = d.source
        a["description"] = d.description
        a["samples"] = len(t)
        rows.append((name, d.units, d.kind, d.selector(), d.source, len(t)))
        if not len(t):
            log("Measurement %s (%s): no matching data in this file." % (d.name, d.selector()))
    st = h5py.string_dtype("utf-8")
    out.create_dataset("index", data=np.array(rows, dtype=[
        ("name", st), ("units", st), ("type", st), ("source", st), ("defined_in", st), ("samples", "u8")]))
    found = sum(1 for r in rows if r[-1])
    log("Measurements: %d defined, %d found in this file." % (len(rows), found))
    return len(rows)
