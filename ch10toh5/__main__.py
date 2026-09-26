"""Run with no arguments for the GUI, or pass files to convert from the command line."""

import argparse
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        from .gui import main as gui_main
        return gui_main()

    from .converter import convert, convert_many, default_output
    ap = argparse.ArgumentParser(prog="ch10toh5", description="Convert Chapter 10 files to HDF5.")
    ap.add_argument("inputs", nargs="+", help="Chapter 10 files (.ch10 / .c10 / .tmt)")
    ap.add_argument("-o", "--output", help="output .h5 path (default: <first input>_combined.h5, "
                                            "or <input>.h5 for a single file)")
    ap.add_argument("--separate", action="store_true", help="write one .h5 per input instead of one combined file")
    ap.add_argument("--year", type=int, help="year for day-of-year time packets")
    ap.add_argument("--no-compress", action="store_true", help="write uncompressed datasets")
    ap.add_argument("--defs", help="CSV of measurement definitions (1553 / ARINC-429 / PCM)")
    args = ap.parse_args(argv)
    opts = dict(year=args.year, compress=not args.no_compress, log=print, definitions=args.defs)
    if args.separate:
        if args.output:
            ap.error("--output cannot be combined with --separate")
        for path in args.inputs:
            convert(path, default_output([path]), **opts)
    else:
        out = args.output or default_output(args.inputs)
        convert_many(args.inputs, out, **opts)
        print("Wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
