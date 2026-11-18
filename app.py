"""Band Stabilizer: small macOS window around stabilize_core."""
import os
import queue
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
        self.root, self.src, self.worker = root, None, None
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

        self.bar = ttk.Progressbar(f, maximum=1.0, length=440)
        self.bar.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.status = ttk.Label(f, text="Choose a video to begin.", wraplength=440)
        self.status.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

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
            self.bar["value"] = 0
            self.status.config(text="Ready. Output: " + os.path.basename(core.output_path(p))
                               if core else "Dependencies missing.")

    def start(self):
        if self.busy() or not self.src or not core:
            return
        dst = core.output_path(self.src)
        if os.path.exists(dst) and not messagebox.askyesno(
                "File exists", f"{os.path.basename(dst)} already exists.\n\nReplace it?"):
            return
        self.cancel.clear()
        for w in (self.go, self.combo):
            w.config(state="disabled")
        args = (self.src, dst, self.strength.get())
        self.worker = threading.Thread(target=self.run, args=args, daemon=True)
        self.worker.start()

    def run(self, src, dst, strength):
        try:
            core.stabilize(src, dst, strength,
                           lambda f, m: self.q.put(("progress", f, m)), self.cancel)
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
