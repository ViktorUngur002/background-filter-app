import mediapipe as mp
import cv2
import numpy as np


class PersonSegmenter:
    def __init__(self):
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

        self.prev_gray = None
        self.prev_mask = None
        self.motion_score = 0.0

        self.base_alpha = 0.2

        # KERNELS
        # Standard noise cleanup
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Connection kernels
        self.kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # Massive glue for high motion
        self.kernel_heavy_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

        # Search area kernels
        self.kernel_search_static = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self.kernel_search_moving = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))

    def _get_skin_mask(self, frame_rgb, is_moving):
        frame_ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)

        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(frame_ycrcb, lower_skin, upper_skin)

        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel_small)

        if is_moving:
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel_heavy_connect)
            skin_mask = cv2.dilate(skin_mask, self.kernel_connect, iterations=1)
        else:
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel_connect)
            skin_mask = cv2.dilate(skin_mask, self.kernel_small, iterations=1)

        return skin_mask.astype(np.float32) / 255.0

    def get_mask(self, frame_rgb):
        mp_result = self.segmenter.process(frame_rgb)
        mp_mask = mp_result.segmentation_mask.astype(np.float32)

        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.prev_gray is not None:
            diff = cv2.absdiff(frame_gray, self.prev_gray)
            self.motion_score = np.sum(diff) / (diff.size * 255)
        else:
            self.motion_score = 0.0

        is_moving = self.motion_score > 0.02

        skin_mask = self._get_skin_mask(frame_rgb, is_moving)

        _, seed_mask = cv2.threshold(mp_mask, 0.5, 1.0, cv2.THRESH_BINARY)
        seed_mask = seed_mask.astype(np.uint8)

        if is_moving:
            search_area = cv2.dilate(seed_mask, self.kernel_search_moving, iterations=3)
        else:
            search_area = cv2.dilate(seed_mask, self.kernel_search_static, iterations=3)

        valid_skin = cv2.bitwise_and(skin_mask, skin_mask, mask=search_area)

        combined_mask = np.maximum(mp_mask, valid_skin)

        current_alpha = self.base_alpha
        if self.motion_score > 0.05:
            current_alpha = 0.0

        if self.prev_mask is not None:
            final_mask = (combined_mask * (1.0 - current_alpha)) + (self.prev_mask * current_alpha)
        else:
            final_mask = combined_mask

        _, output_binary = cv2.threshold(final_mask, 0.5, 1.0, cv2.THRESH_BINARY)

        self.prev_gray = frame_gray
        self.prev_mask = final_mask

        return np.expand_dims(output_binary, axis=-1)

    def get_motion_score(self):
        return self.motion_score