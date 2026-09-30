"""Engineering-unit measurements: turn raw 1553 words, ARINC-429 words and PCM
frame words into named time/value arrays.

Definitions come from two places:

* a CSV file (usually built from the ICD), one row per measurement:
  ``name,type,channel,rt,tr,sa,word,words,label,sdi,bus,word_interval,end_word,
  frame,frame_interval,end_frame,sfid_word,sfid_lsb,sfid_bits,sfid_first,frames,
  lsb,bits,encoding,scale,offset,coefficients,units,description,rate``
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
    word_interval: int = 0          # PCM supercommutation: repeat every N words within the minor frame
    end_word: int = None            # PCM: last word position of a supercommutated word (blank = frame end)
    frame: int = None               # PCM subcommutation: minor frame number in the major frame (1 = first)
    frame_interval: int = 0         # PCM: repeat every N minor frames (0 = only in `frame`)
    end_frame: int = None           # PCM: last minor frame (blank = end of the major frame)
    locations: list = field(default_factory=list)      # PCM: several (word, frame) locations, from TMATS
    sfid: dict = field(default_factory=dict)           # PCM: subframe ID counter settings from the CSV
    lsb: int = None                 # bit field: lowest bit (0 = least significant)
    bits: int = None                # bit field: width
    encoding: str = "unsigned"
    scale: float = 1.0
    offset: float = 0.0
    coefficients: list = field(default_factory=list)   # polynomial a0 + a1*x + a2*x^2 ...
    pairs: list = field(default_factory=list)          # (raw, eu) table, linear interpolation
    units: str = ""
    description: str = ""
    rate: float = None              # expected sample rate in Hz, if the config gives one
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
            if self.end_word:
                s += " to %d" % self.end_word
            if self.frame is not None:
                s += " frame %d" % self.frame
                if self.frame_interval:
                    s += " every %d" % self.frame_interval
                if self.end_frame:
                    s += " to %d" % self.end_frame
            if len(self.locations) > 1:
                s += " (+%d more locations)" % (len(self.locations) - 1)
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


def _channel(v):
    """Channel id: a number, blank (every channel), or a data source name from the TMATS."""
    v = (v or "").strip()
    if not v or v.lower() in ("*", "all", "any"):
        return None
    try:
        return int(v, 0)
    except ValueError:
        return v


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
                    name=row["name"].strip(), kind=kind, channel=_channel(row.get("channel")),
                    rt=_int(row.get("rt")), tr=_int(row.get("tr")), sa=_int(row.get("sa")),
                    word=_int(row.get("word")), words=_int(row.get("words")) or 1,
                    label=_int(row.get("label"), octal=True), sdi=_int(row.get("sdi")), bus=_int(row.get("bus")),
                    word_interval=_int(row.get("word_interval")) or 0, end_word=_int(row.get("end_word")),
                    frame=_int(row.get("frame") or row.get("minor_frame") or row.get("subframe")),
                    frame_interval=_int(row.get("frame_interval")) or 0, end_frame=_int(row.get("end_frame")),
                    sfid={k: _int(row.get(c)) for k, c in (("word", "sfid_word"), ("lsb", "sfid_lsb"),
                          ("bits", "sfid_bits"), ("first_value", "sfid_first"), ("frames", "frames"))
                          if _int(row.get(c)) is not None},
                    lsb=_int(row.get("lsb")), bits=_int(row.get("bits")),
                    encoding=(row.get("encoding", "").strip().lower() or "unsigned"),
                    scale=_float(row.get("scale"), 1.0), offset=_float(row.get("offset"), 0.0),
                    coefficients=[float(c) for c in re.split(r"[;\s]+", row.get("coefficients", "").strip()) if c],
                    units=row.get("units", "").strip(), description=row.get("description", "").strip(),
                    rate=_float(row.get("rate") or row.get("rate_hz") or row.get("frequency"), None))
                if d.encoding not in ENCODINGS:
                    raise ValueError("encoding must be one of " + ", ".join(ENCODINGS))
                if kind == "1553" and None in (d.rt, d.sa, d.word):
                    raise ValueError("1553 rows need rt, sa and word")
                if kind == "arinc429" and d.label is None:
                    raise ValueError("arinc429 rows need a label")
                if kind == "pcm" and d.word is None:
                    raise ValueError("pcm rows need a word")
                if kind == "pcm" and d.frame is not None and d.frame < 1:
                    raise ValueError("frame counts from 1 (the first minor frame of the major frame)")
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
                "frames": int(tm.get("P-%s\\MF\\N" % p, "1") or 1),
            }
            fmt["sfid"] = _tmats_sfid(tm, p, fmt)
        except (KeyError, ValueError):
            continue
        out[int(v)] = fmt
    return out


def _tmats_sfid(tm, p, fmt):
    """Subframe ID counter from P-d\\IDC keys: which word holds it and how it maps to minor frames."""
    def get(n, default=None):
        v = tm.get("P-%s\\IDC%d-1" % (p, n), tm.get("P-%s\\IDC%d" % (p, n), ""))
        v = v.strip()
        return v if v else default
    word = get(1)
    if word is None:
        return None
    wlen = int(get(2, fmt["word_bits"]))
    msb = int(get(3, 1))                 # 1 = the word's most significant bit
    bits = int(get(4, wlen))
    first = int(get(6, 0))
    end = get(8)
    frames = fmt["frames"]
    if frames <= 1 and end is not None:
        frames = abs(int(end) - first) + 1
    return {"word": int(word), "lsb": wlen - (msb - 1) - bits, "bits": bits, "first_value": first,
            "frames": frames, "direction": -1 if get(10, "INC").upper().startswith("DEC") else 1}


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

        def loc(key, m, x=x, y=y, n=n):
            # location m, first fragment; older TMATS leaves out the fragment index
            v = tm.get("D-%s\\%s-%s-%s-%d-1" % (x, key, y, n, m), tm.get("D-%s\\%s-%s-%s-%d" % (x, key, y, n, m)))
            v = (v or "").strip()
            return int(v) if v else None

        nloc = int(tm.get("D-%s\\MML\\N-%s-%s" % (x, y, n), "1") or 1)
        if cid is None or loc("WP", 1) is None or lt != "WDFR":
            log("TMATS measurement %s skipped (location type %s or no PCM link)." % (name, lt))
            continue
        if any(int(tm.get("D-%s\\MNF\\N-%s-%s-%d" % (x, y, n, m), "1") or 1) > 1 for m in range(1, nloc + 1)):
            log("TMATS measurement %s skipped: measurements split across several words are not decoded yet."
                % name)
            continue
        locations = []
        for m in range(1, nloc + 1):
            if loc("WP", m) is None:
                continue
            locations.append(dict(word=loc("WP", m), word_interval=loc("WI", m) or 0, end_word=loc("EWP", m),
                                  frame=loc("FP", m), frame_interval=loc("FI", m) or 0,
                                  end_frame=loc("EFP", m)))
        first = locations[0]
        d = Definition(name=name, kind="pcm", channel=cid, source="tmats", locations=locations, **first)
        mask = tm.get("D-%s\\WFM-%s-%s-1-1" % (x, y, n), tm.get("D-%s\\WFM-%s-%s-1" % (x, y, n), "FW"))
        mask = mask.strip().upper()
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
    """One series per recorder channel and transfer direction (R / T)."""
    series = []
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
        valid = sa_ok & (m["message_error"] == 0) & ((idx + d.words) * 2 <= m["data_length"])
        # the defined RT receives (BC->RT, or receiving side of RT->RT) / transmits
        direction = {"R": valid & ((normal & (m["transmit"] == 0)) | rx),
                     "T": valid & ((normal & (m["transmit"] == 1)) | tx)}
        for dname, ok in direction.items():
            if not ok.any():
                continue
            starts = (m["data_offset"][ok] + 2 * idx[ok]).astype(np.int64)
            b = _gather(g["body"], starts, 2 * d.words)
            w = b.view("<u2").astype(np.uint64)
            raw = np.zeros(len(starts), dtype=np.uint64)
            for k in range(d.words):
                raw = (raw << np.uint64(16)) | w[:, k]
            label = "ch%04d_RT%d_%s_SA%d_W%d" % (int(g.attrs["channel_id"]), d.rt, dname, d.sa, d.word)
            series.append((label, m["time_ns"][ok], raw))
    return series, 0, 16 * d.words


def extract_arinc(root, d):
    """One series per recorder channel and ARINC bus."""
    series = []
    for g in _channels(root, 0x38, d.channel):
        if "messages" not in g or not len(g["messages"]):
            continue
        m = g["messages"].fields(["label_octal", "sdi", "bus", "word", "time_ns", "parity_error", "format_error"])[:]
        ok = (m["label_octal"] == d.label) & (m["format_error"] == 0)
        if d.sdi is not None:
            ok &= m["sdi"] == d.sdi
        if d.bus is not None:
            ok &= m["bus"] == d.bus
        for bus in np.unique(m["bus"][ok]):
            sel = ok & (m["bus"] == bus)
            label = "ch%04d_bus%d_L%03d" % (int(g.attrs["channel_id"]), int(bus), d.label)
            series.append((label, m["time_ns"][sel], m["word"][sel].astype(np.uint64)))
    return series, 10, 29      # default field: data bits 11-29 (19 bits incl. sign)


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


def _word_values(bits, fmt, wp):
    """Bit offset and value of word position wp (1 = sync) in every minor frame, or None if outside."""
    b0 = 0 if wp == 1 else fmt["sync_bits"] + (wp - 2) * fmt["word_bits"]
    wl = fmt["sync_bits"] if wp == 1 else fmt["word_bits"]
    if wp < 1 or b0 + wl > bits.shape[1]:
        return b0, None
    weights = np.uint64(1) << np.arange(wl - 1, -1, -1, dtype=np.uint64)
    return b0, (bits[:, b0:b0 + wl].astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def pcm_locations(d, fmt, frames_per_major):
    """[(word positions, minor frame numbers or None for every frame)] for one PCM definition.

    Supercommutation: a word repeated within the minor frame (word_interval, end_word).
    Subcommutation: a word present only in some minor frames of the major frame
    (frame, frame_interval, end_frame), numbered from 1.
    """
    locs = d.locations or [dict(word=d.word, word_interval=d.word_interval, end_word=d.end_word,
                                frame=d.frame, frame_interval=d.frame_interval, end_frame=d.end_frame)]
    out = []
    for L in locs:
        wi = L.get("word_interval") or 0
        words = list(range(L["word"], (L.get("end_word") or fmt["words"]) + 1, wi)) if wi else [L["word"]]
        f, fi, ef = L.get("frame"), L.get("frame_interval") or 0, L.get("end_frame")
        n = frames_per_major or 0
        if f is None or (f == 1 and fi <= 1 and not ef and n <= 1):
            frames = None                                    # every minor frame
        elif fi:
            frames = list(range(f, (ef or n or f) + 1, fi))
            if n > 1 and frames == list(range(1, n + 1)):
                frames = None
        else:
            frames = [f]
        out.append((words, frames))
    return out


def _sfid(d, fmt):
    s = dict(fmt.get("sfid") or {})
    s.update(d.sfid or {})
    if "word" not in s:
        return None
    wl = fmt["sync_bits"] if s["word"] == 1 else fmt["word_bits"]
    s.setdefault("lsb", 0)
    s.setdefault("bits", wl - s["lsb"])
    s.setdefault("first_value", 0)
    s.setdefault("direction", 1)
    if not s.get("frames") or s["frames"] <= 1:
        s["frames"] = fmt.get("frames") if fmt.get("frames", 1) > 1 else 1 << s["bits"]
    return s


def _frame_numbers(bits, fmt, s):
    """Minor frame number (1 = first of the major frame) of every frame, from the subframe ID counter."""
    _b0, v = _word_values(bits, fmt, s["word"])
    if v is None:
        return None
    v = ((v >> np.uint64(s["lsb"])) & np.uint64((1 << s["bits"]) - 1)).astype(np.int64)
    return (v - s["first_value"]) * s["direction"] % s["frames"] + 1


def extract_pcm(root, defs, formats, log):
    """All PCM definitions for one channel, sharing one pass over its frames."""
    results = {id(d): ([], []) for d in defs}
    cid = defs[0].channel
    fmt = formats.get(cid)
    empty = {id(d): (np.zeros(0, np.int64), np.zeros(0, np.uint64)) for d in defs}
    if fmt is None:
        if any(True for _g in _channels(root, 0x09, cid)):
            log("PCM channel %s: no frame format in TMATS, so its measurements are skipped." % cid)
        return empty, fmt
    plans = {}
    for d in defs:
        s = _sfid(d, fmt)
        plan = pcm_locations(d, fmt, s["frames"] if s else fmt.get("frames", 1))
        if s is None and any(fr is not None for _w, fr in plan):
            log("Measurement %s (%s) is subcommutated, but channel %s has no subframe ID counter "
                "(TMATS P-d\\IDC keys, or the sfid_word column), so it is skipped." % (d.name, d.selector(), cid))
            plan = []
        plans[id(d)] = (s, plan)
    for g in _channels(root, 0x09, cid):
        pk = g["packets"]
        if not len(pk):
            continue
        csdw = {k: int(pk["csdw_" + k][0]) for k in ("throughput", "unpacked", "alignment_32bit")}
        has_frames = "messages" in g and len(g["messages"])
        frames = _frames_iph(g, fmt, csdw) if has_frames else _frames_throughput(g, fmt, csdw)
        ns_per_bit = 1e9 / fmt["bit_rate"] if fmt["bit_rate"] else 0.0
        for t, bits in frames:
            fnums = {}
            for d in defs:
                s, plan = plans[id(d)]
                for words, frame_set in plan:
                    if frame_set is None:
                        sel = slice(None)
                    else:
                        key = tuple(sorted(s.items()))
                        if key not in fnums:
                            fnums[key] = _frame_numbers(bits, fmt, s)
                        if fnums[key] is None:
                            continue
                        sel = np.isin(fnums[key], frame_set)
                    for wp in words:
                        b0, word = _word_values(bits, fmt, wp)
                        if word is None:
                            continue
                        results[id(d)][0].append(t[sel] + int(b0 * ns_per_bit))
                        results[id(d)][1].append(word[sel])
    out = {}
    for d in defs:
        out[id(d)] = _join(*results[id(d)])
    return out, fmt


# ---------------------------------------------------------------- writer

RESERVED = {"raw", "sources", "measurement_index"}


def _safe(name, used):
    base = re.sub(r"[/\\.]+", "_", name).strip() or "measurement"
    if base in RESERVED:
        base += "_m"
    n, k = base, 2
    while n in used:
        n, k = "%s_%d" % (base, k), k + 1
    used.add(n)
    return n


def observed_rate(t):
    """Average sample rate in Hz while data is flowing.

    Spacings longer than max(1 s, 20 x the median spacing) are treated as recording
    gaps and left out. An average rather than the median, so that a supercommutated
    word (several samples bunched in each minor frame) reports its true rate.
    """
    t = t[t >= 0]
    if len(t) < 2:
        return 0.0
    dt = np.diff(np.sort(t)).astype(np.float64)
    if not (dt > 0).any():
        return 0.0
    dt = dt[dt <= max(1e9, 20 * np.median(dt[dt > 0]))]
    return float(len(dt) * 1e9 / dt.sum()) if dt.sum() > 0 else 0.0


def combine(series):
    """Merge the per-source series of one measurement into a single time-ordered series.

    A sample is dropped when another source already has a sample less than
    half a sample period earlier: the same data recorded twice (for example
    the same bus on two recorder channels) is kept once. Returns
    (time_ns, value, raw, duplicates_removed).
    """
    if not series:
        z = np.zeros(0)
        return z.astype(np.int64), z, z.astype(np.int64), 0
    t = np.concatenate([x[2] for x in series]).astype(np.int64)
    v = np.concatenate([x[4] for x in series]).astype(np.float64)
    r = np.concatenate([x[3] for x in series]).astype(np.int64)
    src = np.concatenate([np.full(len(x[2]), k) for k, x in enumerate(series)])
    order = np.lexsort((src, t))
    t, v, r, src = t[order], v[order], r[order], src[order]
    if len(series) > 1 and len(t) > 1:
        rates = [observed_rate(x[2]) for x in series]
        periods = [1e9 / q for q in rates if q > 0]
        tol = 0.5 * min(periods) if periods else 0.0
        dup = np.zeros(len(t), dtype=bool)
        dup[1:] = (src[1:] != src[:-1]) & (np.diff(t) <= tol)
        keep = ~dup
        return t[keep], v[keep], r[keep], int(dup.sum())
    return t, v, r, 0


def resolve_channels(defs, tmats_pairs, log):
    """Allow the channel column to name a data source from the TMATS instead of a number."""
    tm = _tm_dict(tmats_pairs)
    by_name = {}
    for k, v in tm.items():
        m = re.match(r"^R-(\d+)\\TK1-(\d+)$", k)
        if m:
            dsi = tm.get("R-%s\\DSI-%s" % m.groups(), "").strip().lower()
            if dsi:
                by_name[dsi] = int(v)
    for d in defs:
        if isinstance(d.channel, str):
            cid = by_name.get(d.channel.strip().lower())
            if cid is None:
                log("Measurement %s: channel %r is not a number or a data source in the TMATS." % (d.name, d.channel))
            d.channel = cid if cid is not None else -1


def write_measurements(src, out, defs, tmats_pairs, comp, log):
    """Evaluate every definition against the raw data in src and write the results to out.

    Every measurement name becomes a top-level group ``out/<name>`` whose
    ``time_ns``/``value``/``raw`` combine every place the data was found (all
    channels when the channel column is blank, and every row when a name is
    defined more than once), in time order with duplicate copies removed. Each
    place is also kept on its own in ``out/<name>/<source>/`` with its real
    time stamps.
    """
    all_defs = list(defs)
    formats = pcm_formats(tmats_pairs) if tmats_pairs else {}
    if tmats_pairs:
        from_tm = tmats_definitions(tmats_pairs, log)
        if from_tm:
            log("TMATS defines %d PCM measurements." % len(from_tm))
        all_defs += from_tm
    if not all_defs:
        return 0

    resolve_channels(all_defs, tmats_pairs or [], log)
    pcm_channels = sorted(int(g.attrs["channel_id"]) for g in _channels(src, 0x09, None))
    pcm_by_channel = {}
    for d in all_defs:
        if d.kind == "pcm":
            for cid in ([d.channel] if d.channel is not None else pcm_channels):
                pcm_by_channel.setdefault(cid, []).append(d)
    pcm_results = {}          # (id(definition), channel) -> (t, raw)
    for cid, group in pcm_by_channel.items():
        shadow = [Definition(**{**d.__dict__, "channel": cid}) for d in group]
        res, _fmt = extract_pcm(src, shadow, formats, log)
        for d, sd in zip(group, shadow):
            pcm_results[(id(d), cid)] = res[id(sd)]

    by_name = {}
    for d in all_defs:
        by_name.setdefault(d.name, []).append(d)

    st = h5py.string_dtype("utf-8")
    used, rows = set(RESERVED), []
    for mname, mdefs in by_name.items():
        series = []            # (label, definition, t, field_raw, value)
        for k, d in enumerate(mdefs):
            if d.kind == "1553":
                found, lsb0, width = extract_1553(src, d)
            elif d.kind == "arinc429":
                found, lsb0, width = extract_arinc(src, d)
            else:
                found, lsb0, width = [], 0, 16
                for (did, cid), (t, raw) in pcm_results.items():
                    if did == id(d) and len(t):
                        fmt = formats.get(cid) or {}
                        width = fmt.get("word_bits", 16) if d.word != 1 else fmt.get("sync_bits", 16)
                        found.append(("ch%04d_W%d" % (cid, d.word), t, raw))
            if not found:
                log("Measurement %s (%s): no matching data in this file." % (mname, d.selector()))
            for label, t, raw in found:
                if len(mdefs) > 1:
                    label = "def%d_%s" % (k + 1, label)
                order = np.argsort(t, kind="stable")
                t, raw = t[order], raw[order]
                field_raw = _field(raw, d, lsb0, width)
                series.append((label, d, t, field_raw, to_eu(field_raw, d)))

        gname = _safe(mname, used)
        g = out.create_group(gname)
        first = mdefs[0]
        t, value, raw, dups = combine(series)
        kw = comp if len(t) > 1024 else {}
        g.create_dataset("time_ns", data=t, **kw)
        g.create_dataset("value", data=value, **kw)
        g.create_dataset("raw", data=raw, **kw)
        rate = observed_rate(t)
        a = g.attrs
        a["name"] = mname
        a["units"] = first.units
        a["description"] = first.description
        a["type"] = first.kind
        a["definitions"] = len(mdefs)
        a["samples"] = len(t)
        a["rate_hz_observed"] = rate
        a["sources"] = [x[0] for x in series]
        a["duplicates_removed"] = dups
        warn = ""
        cfg_rate = next((d.rate for d in mdefs if d.rate), None)
        if cfg_rate:
            a["rate_hz_config"] = cfg_rate
            if rate and abs(rate - cfg_rate) > 0.2 * cfg_rate:
                warn = ("config says %g Hz but the data arrives at %.3g Hz; samples are written at the "
                        "real rate with their own time stamps" % (cfg_rate, rate))
                a["rate_warning"] = warn
                log("Measurement %s: %s." % (mname, warn))
        if not series:
            a["note"] = "No matching data in this file."
        labels = set()
        for label, d, st_, field_raw, val in series:
            base, n = label, 2
            while label in labels:
                label, n = "%s_%d" % (base, n), n + 1
            labels.add(label)
            s_ = g.create_group(label)
            kw = comp if len(st_) > 1024 else {}
            s_.create_dataset("time_ns", data=st_.astype(np.int64), **kw)
            s_.create_dataset("value", data=val.astype(np.float64), **kw)
            s_.create_dataset("raw", data=field_raw.astype(np.int64), **kw)
            b = s_.attrs
            b["units"] = d.units
            b["source"] = d.selector()
            b["defined_in"] = d.source
            b["samples"] = len(st_)
            b["rate_hz_observed"] = observed_rate(st_)
        if len(series) > 1:
            log("Measurement %s: combined %d sources (%s), %d duplicate samples removed." % (
                mname, len(series), ", ".join(x[0] for x in series), dups))
        rows.append((gname, first.units, first.kind, "; ".join(x[0] for x in series), len(series),
                     len(t), rate, cfg_rate or 0.0, dups, warn))

    out.create_dataset("measurement_index", data=np.array(rows, dtype=[
        ("name", st), ("units", st), ("type", st), ("sources", st), ("source_count", "u4"), ("samples", "u8"),
        ("rate_hz_observed", "f8"), ("rate_hz_config", "f8"), ("duplicates_removed", "u8"), ("warning", st)]))
    found = sum(1 for r in rows if r[5])
    log("Measurements: %d defined, %d found in this file." % (len(by_name), found))
    return len(by_name)
