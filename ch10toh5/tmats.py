"""Minimal TMATS (IRIG 106 Chapter 9) attribute parser."""


def parse_tmats(text):
    """Return (key, value) pairs in file order from TMATS text.

    TMATS attributes look like ``G\\DSI-1:Name;``. Values may span lines, so
    the text is split on semicolons rather than on newlines.
    """
    pairs = []
    for stmt in text.split(";"):
        stmt = stmt.strip()
        if not stmt or ":" not in stmt:
            continue
        key, value = stmt.split(":", 1)
        key = key.strip()
        if key.startswith("COMMENT") or not key:
            continue
        pairs.append((key, value.strip()))
    return pairs
