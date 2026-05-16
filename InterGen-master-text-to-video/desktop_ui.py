import os
import re
import threading
import time
import tkinter as tk
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

import cv2
import torch
from PIL import Image, ImageTk

from configs import get_config
from tools.infer import LitGenModel, build_models


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


class InterGenDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("InterGen Local Motion Generator")
        self.root.geometry("980x760")
        self.root.minsize(860, 680)
        self.root.configure(bg="#f5f6f8")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.loading = True
        self.generating = False
        self.current_video = None
        self.video_cap = None
        self.video_photo = None
        self.playback_job = None

        self._build_ui()
        self._set_status("Loading model on %s..." % self.device)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#111827", height=86)
        header.pack(fill="x")
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="InterGen",
            bg="#111827",
            fg="#ffffff",
            font=("Segoe UI", 24, "bold"),
        )
        title.pack(side="left", padx=(28, 10), pady=18)

        subtitle = tk.Label(
            header,
            text="Text to two-person interactive motion",
            bg="#111827",
            fg="#cbd5e1",
            font=("Segoe UI", 12),
        )
        subtitle.pack(side="left", pady=27)

        main = tk.Frame(self.root, bg="#f5f6f8")
        main.pack(fill="both", expand=True, padx=24, pady=20)

        input_frame = tk.Frame(main, bg="#f5f6f8")
        input_frame.pack(fill="x")

        prompt_label = tk.Label(
            input_frame,
            text="Prompt",
            bg="#f5f6f8",
            fg="#111827",
            font=("Segoe UI", 11, "bold"),
        )
        prompt_label.pack(anchor="w")

        prompt_row = tk.Frame(input_frame, bg="#f5f6f8")
        prompt_row.pack(fill="x", pady=(6, 10))

        self.prompt_var = tk.StringVar(
            value="Two people are dancing together then one falls down."
        )
        self.prompt_entry = tk.Entry(
            prompt_row,
            textvariable=self.prompt_var,
            font=("Segoe UI", 12),
            relief="solid",
            bd=1,
        )
        self.prompt_entry.pack(side="left", fill="x", expand=True, ipady=9)
        self.prompt_entry.bind("<Return>", lambda _event: self.generate())

        self.generate_button = tk.Button(
            prompt_row,
            text="Generate",
            command=self.generate,
            width=14,
            state="disabled",
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
        )
        self.generate_button.pack(side="left", padx=(12, 0), ipady=7)

        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            input_frame,
            textvariable=self.status_var,
            bg="#f5f6f8",
            fg="#475569",
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.status_label.pack(fill="x", pady=(0, 12))

        video_shell = tk.Frame(main, bg="#111827", bd=0, relief="flat")
        video_shell.pack(fill="both", expand=True)

        self.video_label = tk.Label(
            video_shell,
            text="Generated motion will play here",
            bg="#0b1120",
            fg="#94a3b8",
            font=("Segoe UI", 15),
        )
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        controls = tk.Frame(main, bg="#f5f6f8")
        controls.pack(fill="x", pady=(12, 0))

        self.replay_button = tk.Button(
            controls,
            text="Replay",
            command=self.replay,
            state="disabled",
            width=12,
            bg="#e5e7eb",
            fg="#111827",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.replay_button.pack(side="left")

        self.open_folder_button = tk.Button(
            controls,
            text="Open Results Folder",
            command=self.open_results_folder,
            width=20,
            bg="#e5e7eb",
            fg="#111827",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.open_folder_button.pack(side="left", padx=(10, 0))

    def _set_status(self, text):
        self.status_var.set(text)

    def _load_model(self):
        try:
            os.chdir(ROOT)
            model_cfg = get_config("configs/model.yaml")
            infer_cfg = get_config("configs/infer.yaml")
            model = build_models(model_cfg)

            if model_cfg.CHECKPOINT:
                checkpoint_path = ROOT / model_cfg.CHECKPOINT
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                for key in list(ckpt["state_dict"].keys()):
                    if "model" in key:
                        ckpt["state_dict"][key.replace("model.", "")] = ckpt[
                            "state_dict"
                        ].pop(key)
                model.load_state_dict(ckpt["state_dict"], strict=False)

            self.generator = LitGenModel(model, infer_cfg).to(self.device)
            self.generator.eval()
            self.root.after(0, self._model_ready)
        except Exception as exc:
            self.root.after(0, lambda: self._model_failed(exc))

    def _model_ready(self):
        self.loading = False
        self.generate_button.configure(state="normal")
        self._set_status("Model loaded. Type a prompt and press Generate.")

    def _model_failed(self, exc):
        self.loading = False
        self._set_status("Model failed to load.")
        messagebox.showerror("InterGen", "Could not load the model:\n\n%s" % exc)

    def generate(self):
        if self.loading or self.generating:
            return

        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showinfo("InterGen", "Please enter a prompt first.")
            return

        self.generating = True
        self.generate_button.configure(state="disabled", text="Generating...")
        self.replay_button.configure(state="disabled")
        self._stop_video()
        self.video_label.configure(
            image="",
            text="Generating motion...\nThis usually takes about a minute.",
            fg="#e5e7eb",
        )
        self._set_status("Generating: %s" % prompt)
        threading.Thread(target=self._generate_video, args=(prompt,), daemon=True).start()

    def _generate_video(self, prompt):
        try:
            RESULTS_DIR.mkdir(exist_ok=True)
            safe_prompt = re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_")[:48]
            if not safe_prompt:
                safe_prompt = "motion"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / ("ui_%s_%s.mp4" % (timestamp, safe_prompt))

            batch = OrderedDict()
            batch["motion_lens"] = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            batch["prompt"] = prompt

            with torch.no_grad():
                motion_output = self.generator.generate_loop(batch, window_size=210)
                self.generator.plot_t2m(
                    [motion_output[0], motion_output[1]],
                    str(output_path),
                    prompt,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.root.after(0, lambda: self._generation_done(output_path))
        except Exception as exc:
            self.root.after(0, lambda: self._generation_failed(exc))

    def _generation_done(self, output_path):
        self.generating = False
        self.generate_button.configure(state="normal", text="Generate")
        self.replay_button.configure(state="normal")
        self.current_video = output_path
        self._set_status("Done: %s" % output_path.name)
        self.play_video(output_path)

    def _generation_failed(self, exc):
        self.generating = False
        self.generate_button.configure(state="normal", text="Generate")
        self._set_status("Generation failed.")
        self.video_label.configure(
            image="",
            text="Generation failed",
            fg="#fecaca",
        )
        messagebox.showerror("InterGen", "Could not generate the video:\n\n%s" % exc)

    def play_video(self, path):
        self._stop_video()
        self.video_cap = cv2.VideoCapture(str(path))
        if not self.video_cap.isOpened():
            messagebox.showerror("InterGen", "Could not open video:\n%s" % path)
            return
        self._show_next_frame()

    def _show_next_frame(self):
        if self.video_cap is None:
            return

        ok, frame = self.video_cap.read()
        if not ok:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.video_cap.read()
            if not ok:
                return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_width = max(self.video_label.winfo_width(), 320)
        label_height = max(self.video_label.winfo_height(), 240)
        frame_height, frame_width = frame.shape[:2]
        scale = min(label_width / frame_width, label_height / frame_height)
        new_size = (max(1, int(frame_width * scale)), max(1, int(frame_height * scale)))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        image = Image.fromarray(frame)
        self.video_photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.video_photo, text="")

        fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(10, int(1000 / fps))
        self.playback_job = self.root.after(delay, self._show_next_frame)

    def _stop_video(self):
        if self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

    def replay(self):
        if self.current_video:
            self.play_video(self.current_video)

    def open_results_folder(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        os.startfile(RESULTS_DIR)

    def close(self):
        self._stop_video()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = InterGenDesktopApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
