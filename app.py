"""Band Stabilizer: small macOS window around stabilize_core."""
import os
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import stabilize_core as core
    IMPORT_ERROR = None
except ImportError as e:  # numpy / opencv missing
    core, IMPORT_ERROR = None, e

VIDEO_TYPES = [("Videos", "*.mov *.MOV *.mp4 *.MP4 *.m4v *.M4V *.avi *.mkv *.mts *.MTS"),
               ("All files", "*.*")]


class App:
    def __init__(self, root):
        self.root, self.src, self.worker, self.region = root, None, None, None
        self.cancel, self.q = threading.Event(), queue.Queue()
        root.title("Band Stabilizer")
        root.resizable(False, False)

        f = ttk.Frame(root, padding=14)
        f.grid(sticky="nsew")
        ttk.Button(f, text="Choose Video…", command=self.choose).grid(row=0, column=0, sticky="w")
        self.file_lbl = ttk.Label(f, text="No video chosen", width=46)
        self.file_lbl.grid(row=0, column=1, columnspan=2, sticky="w", padx=(10, 0))

        ttk.Label(f, text="Strength:").grid(row=1, column=0, sticky="w", pady=10)
        self.strength = tk.StringVar(value="Gentle")
        names = list(core.STRENGTHS) if core else ["Gentle"]
        self.combo = ttk.Combobox(f, textvariable=self.strength, values=names,
                                  state="readonly", width=10)
        self.combo.grid(row=1, column=1, sticky="w", padx=(10, 0))
        self.go = ttk.Button(f, text="Stabilize", command=self.start, state="disabled")
        self.go.grid(row=1, column=2, sticky="e")

        self.ref_btn = ttk.Button(f, text="Set Bleachers Area…", command=self.pick_region,
                                  state="disabled")
        self.ref_btn.grid(row=2, column=0, sticky="w", pady=(0, 10))
        self.ref_lbl = ttk.Label(f, text="Reference: whole frame")
        self.ref_lbl.grid(row=2, column=1, columnspan=2, sticky="w", padx=(10, 0), pady=(0, 10))

        self.bar = ttk.Progressbar(f, maximum=1.0, length=440)
        self.bar.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.status = ttk.Label(f, text="Choose a video to begin.", wraplength=440)
        self.status.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

        menubar = tk.Menu(root)
        fm = tk.Menu(menubar, tearoff=0)
        fm.add_command(label="Open…", accelerator="Command-O", command=self.choose)
        fm.add_separator()
        fm.add_command(label="Quit", accelerator="Command-Q", command=self.quit)
        menubar.add_cascade(label="File", menu=fm)
        root.config(menu=menubar)
        root.bind_all("<Command-o>", lambda e: self.choose())
        root.protocol("WM_DELETE_WINDOW", self.quit)
        try:
            root.createcommand("tk::mac::Quit", self.quit)
        except tk.TclError:
            pass
        root.after(100, self.poll)
        root.after(200, self.check_deps)

    def check_deps(self):
        try:
            if IMPORT_ERROR:
                raise RuntimeError(f"Python package missing ({IMPORT_ERROR.name}).\n\nRun in "
                                   "Terminal from the app folder:\n.venv/bin/pip install "
                                   "opencv-python-headless numpy")
            core.check_dependencies()
        except Exception as e:
            self.status.config(text="Missing dependency — see message.")
            messagebox.showerror("Missing dependency", str(e))

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def choose(self):
        if self.busy():
            return
        p = filedialog.askopenfilename(title="Choose a video", filetypes=VIDEO_TYPES)
        if p:
            self.src = p
            self.file_lbl.config(text=os.path.basename(p))
            self.go.config(state="normal" if core else "disabled")
            self.ref_btn.config(state="normal" if core else "disabled")
            self.bar["value"] = 0
            self.set_region(core.saved_region(p) if core else None)
            self.status.config(text="Tip: click Set Bleachers Area and drag a box over the "
                               "stands/buildings, then Stabilize." if core else
                               "Dependencies missing.")

    def set_region(self, region):
        self.region = region
        self.ref_lbl.config(text="Reference: whole frame" if region is None else
                            "Reference: selected bleachers area ✓")

    def pick_region(self):
        if self.busy() or not self.src:
            return
        png = os.path.join(tempfile.gettempdir(), "band_stabilizer_frame.png")
        try:
            core.first_frame_png(self.src, png)
            img = tk.PhotoImage(file=png)
        except Exception as e:
            messagebox.showerror("Can't show frame", str(e))
            return
        top = tk.Toplevel(self.root)
        top.title("Frame at 3 s — drag a box over the bleachers / stands (not the band)")
        top.transient(self.root)
        iw, ih = img.width(), img.height()
        cv = tk.Canvas(top, width=iw, height=ih, highlightthickness=0, cursor="crosshair")
        cv.pack()
        cv.create_image(0, 0, image=img, anchor="nw")
        cv.image = img
        st = {"rect": None, "x0": 0, "y0": 0, "box": None}
        if self.region:
            x0, y0, x1, y1 = self.region
            st["box"] = (x0 * iw, y0 * ih, x1 * iw, y1 * ih)
            st["rect"] = cv.create_rectangle(*st["box"], outline="#ffd60a", width=3)

        def down(e):
            st["x0"], st["y0"] = e.x, e.y
            if st["rect"]:
                cv.delete(st["rect"])
            st["rect"] = cv.create_rectangle(e.x, e.y, e.x, e.y, outline="#ffd60a", width=3)

        def drag(e):
            x, y = min(max(e.x, 0), iw), min(max(e.y, 0), ih)
            cv.coords(st["rect"], st["x0"], st["y0"], x, y)
            st["box"] = (min(st["x0"], x), min(st["y0"], y), max(st["x0"], x), max(st["y0"], y))

        def use():
            b = st["box"]
            if not b or (b[2] - b[0]) < 0.1 * iw or (b[3] - b[1]) < 0.05 * ih:
                messagebox.showinfo("Area too small", "Drag a larger box over the bleachers.",
                                    parent=top)
                return
            self.set_region((b[0] / iw, b[1] / ih, b[2] / iw, b[3] / ih))
            top.destroy()

        def whole():
            self.set_region(None)
            top.destroy()

        cv.bind("<ButtonPress-1>", down)
        cv.bind("<B1-Motion>", drag)
        bar = ttk.Frame(top, padding=8)
        bar.pack(fill="x")
        ttk.Label(bar, text="Include lots of still detail: stands, railings, buildings, "
                  "goal posts.").pack(side="left")
        ttk.Button(bar, text="Use This Area", command=use).pack(side="right")
        ttk.Button(bar, text="Whole Frame", command=whole).pack(side="right", padx=6)
        top.grab_set()

    def start(self):
        if self.busy() or not self.src or not core:
            return
        dst = core.output_path(self.src)
        if os.path.exists(dst) and not messagebox.askyesno(
                "File exists", f"{os.path.basename(dst)} already exists.\n\nReplace it?"):
            return
        self.cancel.clear()
        for w in (self.go, self.combo, self.ref_btn):
            w.config(state="disabled")
        args = (self.src, dst, self.strength.get(), self.region)
        self.worker = threading.Thread(target=self.run, args=args, daemon=True)
        self.worker.start()

    def run(self, src, dst, strength, region):
        try:
            core.stabilize(src, dst, strength,
                           lambda f, m: self.q.put(("progress", f, m)), self.cancel, region)
            self.q.put(("done", None, None))
        except core.Cancelled:
            self.q.put(("cancelled", None, None))
        except core.StabilizeError as e:
            self.q.put(("error", None, str(e)))
        except Exception as e:  # unexpected
            self.q.put(("error", None, f"Unexpected error: {type(e).__name__}: {e}"))

    def poll(self):
        try:
            while True:
                kind, frac, msg = self.q.get_nowait()
                if kind == "progress":
                    self.bar["value"] = frac
                    self.status.config(text=msg)
                else:
                    self.go.config(state="normal")
                    self.ref_btn.config(state="normal")
                    self.combo.config(state="readonly")
                    if kind == "error":
                        self.status.config(text="Failed — original file untouched.")
                        messagebox.showerror("Stabilization failed", msg)
                    elif kind == "done":
                        self.bar["value"] = 1.0
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def quit(self):
        if self.busy():
            if not messagebox.askyesno("Quit", "Processing is in progress. Stop and quit?"):
                return
            self.cancel.set()
            self.worker.join(timeout=5)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.lift()
    root.after(300, lambda: root.attributes("-topmost", False))
    root.attributes("-topmost", True)
    root.mainloop()
