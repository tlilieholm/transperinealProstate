"""
Thomas Lilieholm, UW Madison Dept. of Medical Physics, 2024

Tkinter front-end for gridV9.battleshipAim(). Replaces manual editing of the
hard-coded coordinates in ProstateGridV9.py with input fields for the three
registration fiducials (ARM, LL, UR) and one or more target points.
"""

import contextlib
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from gridV9 import battleshipAim

AXIS_LABELS = ("LR", "AP", "SI")

TEST_DATA = {
    "ARM": [-0.94, 52.12, -197.74],
    "LL": [-26.57, -37.68, -112.74],
    "UR": [36.20, 41.54, -117.74],
    "targets": [
        [-11.52, -6.88, -32.74],
        [5.87, -7.82, -37.74],
        [-15.05, -2.89, -47.74],
        [8.46, -2.42, -42.74],
        [-18.79, -14.03, -32.74],
        [11.96, -15.48, -32.74],
        [-17.40, -25.22, -32.74],
        [12.69, -25.22, -32.74],
        [-3.53, -28.27, -32.74],
        [7.99, -13.00, -32.74],
    ],
}


class CoordinateFields(ttk.Frame):
    """Three labeled entry boxes (LR, AP, SI) for one 3D point."""

    def __init__(self, parent, error_name):
        super().__init__(parent)
        self.error_name = error_name
        self.vars = []
        for i, axis in enumerate(AXIS_LABELS):
            ttk.Label(self, text=axis).grid(row=0, column=2 * i, padx=(0 if i == 0 else 6, 2))
            var = tk.StringVar()
            ttk.Entry(self, textvariable=var, width=9).grid(row=0, column=2 * i + 1)
            self.vars.append(var)

    def get_point(self):
        values = []
        for axis, var in zip(AXIS_LABELS, self.vars):
            text = var.get().strip()
            if not text:
                raise ValueError(f"{self.error_name}: {axis} is empty.")
            try:
                values.append(float(text))
            except ValueError:
                raise ValueError(f"{self.error_name}: {axis} value '{text}' is not a number.") from None
        return values

    def is_blank(self):
        return all(not var.get().strip() for var in self.vars)

    def set_point(self, point):
        for var, value in zip(self.vars, point):
            var.set(str(value))

    def clear(self):
        for var in self.vars:
            var.set("")


class RegistrationRow(ttk.Frame):
    """One fixed registration fiducial (ARM / LL / UR)."""

    def __init__(self, parent, name, description):
        super().__init__(parent)
        ttk.Label(self, text=name, width=6, font=("", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.fields = CoordinateFields(self, name)
        self.fields.grid(row=0, column=1, padx=(4, 10))
        ttk.Label(self, text=description, foreground="#666").grid(row=0, column=2, sticky="w")

    def get_point(self):
        return self.fields.get_point()

    def set_point(self, point):
        self.fields.set_point(point)

    def clear(self):
        self.fields.clear()


class TargetRow(ttk.Frame):
    """One removable target point row."""

    def __init__(self, parent, index, on_remove):
        super().__init__(parent)
        self._on_remove = on_remove
        self.label = ttk.Label(self, width=10)
        self.label.grid(row=0, column=0, sticky="w")
        self.fields = CoordinateFields(self, "")
        self.fields.grid(row=0, column=1, padx=(4, 10))
        self.remove_btn = ttk.Button(self, text="Remove", width=8, command=lambda: self._on_remove(self))
        self.remove_btn.grid(row=0, column=2)
        self.set_index(index)

    def set_index(self, index):
        self.index = index
        self.label.config(text=f"Target {index}")
        self.fields.error_name = f"Target {index}"

    def get_point(self):
        return self.fields.get_point()

    def is_blank(self):
        return self.fields.is_blank()

    def set_point(self, point):
        self.fields.set_point(point)

    def clear(self):
        self.fields.clear()


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container; add children to .inner."""

    def __init__(self, parent, height=160):
        super().__init__(parent)
        canvas = tk.Canvas(self, height=height, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class TextWidgetWriter:
    """File-like object that appends written text to a Tk Text widget, for redirecting stdout."""

    def __init__(self, text_widget):
        self._widget = text_widget

    def write(self, message):
        if not message:
            return
        self._widget.config(state="normal")
        self._widget.insert("end", message)
        self._widget.see("end")
        self._widget.config(state="disabled")
        self._widget.update_idletasks()

    def flush(self):
        pass


class ProstateGridApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ProstateGrid — Registration & Targeting")
        self.geometry("880x780")
        self.minsize(760, 600)

        self._target_rows = []
        self._build_layout()
        self._add_target_row()

    # ---- Layout ----
    def _build_layout(self):
        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)

        form_container = ttk.Frame(paned, padding=12)
        paned.add(form_container, weight=3)

        results_container = ttk.Frame(paned, padding=(12, 0, 12, 12))
        paned.add(results_container, weight=2)

        self._build_form(form_container)
        self._build_results(results_container)

    def _build_form(self, parent):
        ttk.Label(
            parent,
            text="Coordinates are [LR, AP, SI] in mm. Patient left, anterior, and superior are positive.",
            foreground="#555",
        ).pack(anchor="w", pady=(0, 10))

        reg_frame = ttk.LabelFrame(parent, text="Registration Fiducials", padding=10)
        reg_frame.pack(fill="x", pady=(0, 10))

        self.arm_row = RegistrationRow(reg_frame, "ARM", "Noncoplanar arm fiducial")
        self.arm_row.pack(anchor="w", pady=2)
        self.ll_row = RegistrationRow(reg_frame, "LL", "Lower-left fiducial")
        self.ll_row.pack(anchor="w", pady=2)
        self.ur_row = RegistrationRow(reg_frame, "UR", "Upper-right fiducial")
        self.ur_row.pack(anchor="w", pady=2)

        tar_frame = ttk.LabelFrame(parent, text="Target Points", padding=10)
        tar_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.targets_scroll = ScrollableFrame(tar_frame, height=160)
        self.targets_scroll.pack(fill="both", expand=True)

        ttk.Button(tar_frame, text="+ Add Target", command=self._add_target_row).pack(anchor="w", pady=(8, 0))

        opt_frame = ttk.LabelFrame(parent, text="Options", padding=10)
        opt_frame.pack(fill="x", pady=(0, 10))

        self.show_var = tk.BooleanVar(value=True)
        self.optimize_var = tk.BooleanVar(value=True)
        self.verbose_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Show 3D visualization", variable=self.show_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opt_frame, text="Optimize ARM Z", variable=self.optimize_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opt_frame, text="Verbose output", variable=self.verbose_var).pack(side="left")

        action_frame = ttk.Frame(parent)
        action_frame.pack(fill="x")
        ttk.Button(action_frame, text="Calculate", command=self._on_calculate).pack(side="left")
        ttk.Button(action_frame, text="Load Test Data", command=self._on_load_test_data).pack(side="left", padx=8)
        ttk.Button(action_frame, text="Clear All", command=self._on_clear_all).pack(side="left")

    def _build_results(self, parent):
        ttk.Label(parent, text="Results", font=("", 10, "bold")).pack(anchor="w")
        self.results_text = scrolledtext.ScrolledText(parent, height=14, state="disabled", font=("Consolas", 9))
        self.results_text.pack(fill="both", expand=True, pady=(4, 0))

    # ---- Target row management ----
    def _add_target_row(self):
        row = TargetRow(self.targets_scroll.inner, len(self._target_rows) + 1, self._remove_target_row)
        row.pack(anchor="w", pady=2, fill="x")
        self._target_rows.append(row)

    def _remove_target_row(self, row):
        if len(self._target_rows) == 1:
            messagebox.showinfo("ProstateGrid", "At least one target point is required.")
            return
        self._target_rows.remove(row)
        row.destroy()
        for i, r in enumerate(self._target_rows, start=1):
            r.set_index(i)

    # ---- Data helpers ----
    def _on_load_test_data(self):
        self.arm_row.set_point(TEST_DATA["ARM"])
        self.ll_row.set_point(TEST_DATA["LL"])
        self.ur_row.set_point(TEST_DATA["UR"])

        while len(self._target_rows) < len(TEST_DATA["targets"]):
            self._add_target_row()
        while len(self._target_rows) > len(TEST_DATA["targets"]):
            self._remove_target_row(self._target_rows[-1])

        for row, point in zip(self._target_rows, TEST_DATA["targets"]):
            row.set_point(point)

    def _on_clear_all(self):
        self.arm_row.clear()
        self.ll_row.clear()
        self.ur_row.clear()
        for row in list(self._target_rows[1:]):
            self._remove_target_row(row)
        self._target_rows[0].clear()
        self._set_results("")

    # ---- Calculation ----
    def _collect_targets(self):
        targets = []
        for row in self._target_rows:
            if row.is_blank():
                continue
            targets.append(row.get_point())
        if not targets:
            raise ValueError("At least one target point must be filled in.")
        return targets

    def _on_calculate(self):
        try:
            grid_arm = self.arm_row.get_point()
            grid_ll = self.ll_row.get_point()
            grid_ur = self.ur_row.get_point()
            targets = self._collect_targets()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        log_lines = []
        # Radiologic view correction (mirrors ProstateGridV9.py main())
        if grid_ll[0] > grid_ur[0] and grid_arm[2] > grid_ll[2]:
            log_lines.append("Radiologic View Detected — flipping LR and SI axes.\n")
            grid_ll[0], grid_ll[2] = -grid_ll[0], -grid_ll[2]
            grid_ur[0], grid_ur[2] = -grid_ur[0], -grid_ur[2]
            grid_arm[0], grid_arm[2] = -grid_arm[0], -grid_arm[2]
            for t in targets:
                t[0], t[2] = -t[0], -t[2]

        log_lines.append(f"LL coordinates: {grid_ll}")
        log_lines.append(f"UR coordinates: {grid_ur}")
        log_lines.append(f"ARM coordinates: {grid_arm}")
        for i, t in enumerate(targets, start=1):
            log_lines.append(f"TAR{i} coordinates: {t}")

        self._set_results("\n".join(log_lines) + "\n\n")

        writer = TextWidgetWriter(self.results_text)
        try:
            with contextlib.redirect_stdout(writer):
                battleshipAim(
                    grid_ll,
                    grid_ur,
                    grid_arm,
                    targets,
                    show=self.show_var.get(),
                    optimizeZ=self.optimize_var.get(),
                    verbose=self.verbose_var.get(),
                )
        except Exception:
            self._append_results("\nERROR during calculation:\n" + traceback.format_exc())
            messagebox.showerror(
                "Calculation error",
                "An error occurred during calculation. See the results panel for details.",
            )

    def _set_results(self, text):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", text)
        self.results_text.config(state="disabled")

    def _append_results(self, text):
        self.results_text.config(state="normal")
        self.results_text.insert("end", text)
        self.results_text.see("end")
        self.results_text.config(state="disabled")


def main():
    app = ProstateGridApp()
    app.mainloop()


if __name__ == "__main__":
    main()
