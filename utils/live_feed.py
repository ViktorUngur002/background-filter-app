import cv2
import customtkinter as ctk
from PIL import Image
from processing.segmenter import PersonSegmenter
from processing.background_apply import apply_blur_background, apply_pattern_background

class LiveFeed:
    def __init__(self, root, cap, video_label, get_frame_size_callback):
        self.root = root
        self.cap = cap
        self.video_label = video_label
        self.get_frame_size_callback = get_frame_size_callback

        self.is_paused = False
        self.is_recording = False
        self.recorded_frames = []
        self.after_id = None
        self.last_processed_frame = None

        self.segmenter = PersonSegmenter()
        self.selected_pattern = None
        self.effect_mode = "none"

        self.update_video()

    def set_effect_mode(self, effect_mode):
        self.effect_mode = effect_mode

    def set_selected_pattern(self, selected_pattern):
        self.selected_pattern = selected_pattern

    def get_last_processed_frame(self):
        return self.last_processed_frame

    def start_recording(self):
        self.is_recording = True
        self.recorded_frames = []

    def stop_recording(self):
        self.is_recording = False
        return self.recorded_frames.copy()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def get_frame_size(self):
        return self.get_frame_size_callback()

    def is_lf_recording(self):
        return self.is_recording

    def update_video(self):
        if not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mask = self.segmenter.get_mask(rgb_frame)

                if self.effect_mode == "blur":
                    processed_frame = apply_blur_background(rgb_frame, mask)
                elif self.effect_mode == "pattern" and self.selected_pattern is not None:
                    processed_frame = apply_pattern_background(rgb_frame, mask, self.selected_pattern)
                else:
                    processed_frame = rgb_frame

                self.last_processed_frame = processed_frame.copy()

                img = Image.fromarray(processed_frame)

                w, h = self.get_frame_size()
                img = img.resize((w, h))

                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                self.video_label.configure(image=imgtk)
                self.video_label.imgtk = imgtk

                if self.is_recording:
                    self.recorded_frames.append(processed_frame)

        if self.after_id:
            self.video_label.after_cancel(self.after_id)
        self.after_id = self.video_label.after(30, self.update_video)