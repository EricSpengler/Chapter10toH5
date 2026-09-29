"""Tkinter front end for the Chapter 10 to HDF5 converter."""

import os
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .converter import Cancelled, convert_many, default_output

CH10_TYPES = [("Chapter 10 files", "*.ch10 *.c10 *.tmt *.CH10 *.C10 *.TMT"), ("All files", "*.*")]


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.master = master
        master.title("Chapter 10 to HDF5  v%s" % __version__)
        master.minsize(640, 460)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None

        # Input files
        files = ttk.LabelFrame(self, text="Chapter 10 files", padding=6)
        files.grid(row=0, column=0, sticky="nsew")
        self.listbox = tk.Listbox(files, height=6, selectmode=tk.EXTENDED)
        self.listbox.grid(row=0, column=0, rowspan=3, sticky="nsew")
        sb = ttk.Scrollbar(files, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.grid(row=0, column=1, rowspan=3, sticky="ns")
        self.listbox.configure(yscrollcommand=sb.set)
        ttk.Button(files, text="Add files...", command=self.add_files).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        ttk.Button(files, text="Remove", command=self.remove_selected).grid(row=1, column=2, sticky="ew", padx=(6, 0))
        ttk.Button(files, text="Clear", command=lambda: self.listbox.delete(0, tk.END)).grid(
            row=2, column=2, sticky="new", padx=(6, 0))
        files.columnconfigure(0, weight=1)
        files.rowconfigure(2, weight=1)

        # Options
        opts = ttk.LabelFrame(self, text="Output", padding=6)
        opts.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.outdir = tk.StringVar()
        ttk.Label(opts, text="Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.outdir).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(opts, text="Browse...", command=self.pick_outdir).grid(row=0, column=2)
        ttk.Label(opts, text="(blank = next to the first input file; several inputs go into one .h5)",
                  foreground="gray").grid(
            row=1, column=1, sticky="w", padx=4)
        self.year = tk.StringVar()
        ttk.Label(opts, text="Year:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(opts, textvariable=self.year, width=8).grid(row=2, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(opts, text="only needed when time packets use day-of-year format", foreground="gray").grid(
            row=3, column=1, sticky="w", padx=4)
        self.defs = tk.StringVar()
        ttk.Label(opts, text="Definitions:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(opts, textvariable=self.defs).grid(row=5, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Button(opts, text="Browse...", command=self.pick_defs).grid(row=5, column=2, pady=(6, 0))
        ttk.Label(opts, text="optional ICD CSV (see docs/definitions_template.csv)",
                  foreground="gray").grid(row=6, column=1, sticky="w", padx=4)
        self.include_raw = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Include raw dump (every packet, under /raw)", variable=self.include_raw).grid(
            row=7, column=1, sticky="w", padx=4, pady=(6, 0))
        self.compress = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Compress (gzip)", variable=self.compress).grid(
            row=4, column=1, sticky="w", padx=4, pady=(6, 0))
        opts.columnconfigure(1, weight=1)

        # Run controls
        run = ttk.Frame(self)
        run.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.convert_btn = ttk.Button(run, text="Convert", command=self.start)
        self.convert_btn.grid(row=0, column=0)
        self.cancel_btn = ttk.Button(run, text="Cancel", command=self.cancel.set, state=tk.DISABLED)
        self.cancel_btn.grid(row=0, column=1, padx=6)
        self.status = tk.StringVar(value="Add one or more Chapter 10 files, then press Convert.")
        ttk.Label(run, textvariable=self.status).grid(row=0, column=2, sticky="w")
        self.bar = ttk.Progressbar(run, maximum=1000)
        self.bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        run.columnconfigure(2, weight=1)

        # Log
        logf = ttk.LabelFrame(self, text="Log", padding=6)
        logf.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        self.log = tk.Text(logf, height=10, wrap="word", state=tk.DISABLED)
        self.log.grid(row=0, column=0, sticky="nsew")
        lsb = ttk.Scrollbar(logf, orient=tk.VERTICAL, command=self.log.yview)
        lsb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=lsb.set)
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(3, weight=2)
        self.after(100, self.poll)

    # ------------------------------------------------------------ actions

    def add_files(self):
        for path in filedialog.askopenfilenames(title="Choose Chapter 10 files", filetypes=CH10_TYPES):
            if path not in self.listbox.get(0, tk.END):
                self.listbox.insert(tk.END, path)

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)

    def pick_defs(self):
        path = filedialog.askopenfilename(title="Choose measurement definitions",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.defs.set(path)

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.outdir.set(d)

    def write_log(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def start(self):
        files = list(self.listbox.get(0, tk.END))
        if not files:
            messagebox.showinfo("Nothing to convert", "Add at least one Chapter 10 file first.")
            return
        year = self.year.get().strip()
        if year and not (year.isdigit() and 1970 <= int(year) <= 2200):
            messagebox.showerror("Year", "Year must be a four-digit number, or left blank.")
            return
        defs = self.defs.get().strip() or None
        if defs and not os.path.isfile(defs):
            messagebox.showerror("Definitions", "That definitions file does not exist.")
            return
        if not defs and not self.include_raw.get() and not messagebox.askyesno(
                "Nothing selected", "The raw dump is off and no definitions file is set, so only "
                "measurements defined in the recording's TMATS would be written. Continue?"):
            return
        outdir = self.outdir.get().strip()
        if outdir and not os.path.isdir(outdir):
            messagebox.showerror("Output folder", "That output folder does not exist.")
            return
        out = default_output(files, outdir or None)
        if len(files) > 1:
            out = filedialog.asksaveasfilename(
                title="Save combined HDF5 file", initialdir=os.path.dirname(out),
                initialfile=os.path.basename(out), defaultextension=".h5",
                filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")])
            if not out:
                return
        self.cancel.clear()
        self.convert_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.NORMAL)
        self.worker = threading.Thread(
            target=self.run, args=(files, out, int(year) if year else None, self.compress.get(), defs,
                                   self.include_raw.get()),
            daemon=True)
        self.worker.start()

    def run(self, files, out, year, compress, defs=None, include_raw=True):
        ev = self.events
        ev.put(("status", "Converting %d file%s..." % (len(files), "s" if len(files) > 1 else "")))
        ev.put(("log", "=== %d file%s -> %s" % (len(files), "s" if len(files) > 1 else "", out)))
        try:
            results = convert_many(files, out, year=year, compress=compress, cancel=self.cancel,
                                   definitions=defs, include_raw=include_raw,
                                   progress=lambda d, t: ev.put(("progress", d / t if t else 1.0)),
                                   log=lambda m: ev.put(("log", m)))
            errors = sum(r["decode_errors"] for r in results)
            if errors:
                ev.put(("log", "Note: %d packets could not be decoded into messages; their raw "
                               "bytes are still in the file." % errors))
            if len(results) > 1:
                ev.put(("log", "Each input is a top-level group: " + ", ".join(r["group"] for r in results)))
            ev.put(("done", "Wrote %s" % os.path.basename(out)))
        except Cancelled:
            ev.put(("done", "Cancelled. No file was written."))
        except Exception as exc:
            ev.put(("log", "FAILED: %s\n%s" % (exc, traceback.format_exc())))
            ev.put(("done", "Conversion failed. See the log."))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.write_log(value)
                elif kind == "status":
                    self.status.set(value)
                    self.bar["value"] = 0
                elif kind == "progress":
                    self.bar["value"] = value * 1000
                elif kind == "done":
                    self.status.set(value)
                    self.write_log(value)
                    self.convert_btn.configure(state=tk.NORMAL)
                    self.cancel_btn.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(100, self.poll)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
