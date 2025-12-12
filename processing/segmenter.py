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

        # --- TUNING ---
        self.base_alpha = 0.2

        # --- KERNELS ---
        # Standard noise cleanup
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Connection kernels (The "Glue")
        self.kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # Massive glue for high motion (Fixes slicing)
        self.kernel_heavy_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

        # Search area kernels (The "Reach")
        self.kernel_search_static = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self.kernel_search_moving = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))

    def _get_skin_mask(self, frame_rgb, is_moving):
        """
        Adapts skin detection based on movement.
        """
        # 1. Color Segmentation (YCrCb)
        frame_ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)

        # Standard Skin Range
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(frame_ycrcb, lower_skin, upper_skin)

        # 2. Cleanup
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel_small)

        # 3. --- DYNAMIC REPAIR (The Fix) ---
        if is_moving:
            # MOVEMENT MODE: Aggressive Repair
            # 1. "Close" huge gaps. Motion blur creates holes; this fills them.
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel_heavy_connect)

            # 2. Dilate. Motion blur makes the hand look thin/transparent.
            # We add pixels back to restore volume.
            skin_mask = cv2.dilate(skin_mask, self.kernel_connect, iterations=1)
        else:
            # STATIC MODE: Gentle Repair
            # Just connect fingers, don't blob too much
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel_connect)
            skin_mask = cv2.dilate(skin_mask, self.kernel_small, iterations=1)

        return skin_mask.astype(np.float32) / 255.0

    def get_mask(self, frame_rgb):
        # 1. MediaPipe Body (The "Core")
        mp_result = self.segmenter.process(frame_rgb)
        mp_mask = mp_result.segmentation_mask.astype(np.float32)

        # 2. Motion Analysis
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.prev_gray is not None:
            diff = cv2.absdiff(frame_gray, self.prev_gray)
            self.motion_score = np.sum(diff) / (diff.size * 255)
        else:
            self.motion_score = 0.0

        # Thresholds:
        # > 0.02 means "Moving" (Use aggressive skin glue)
        # > 0.05 means "Fast" (Drop temporal smoothing)
        is_moving = self.motion_score > 0.02

        # 3. Intelligent Skin Recovery
        skin_mask = self._get_skin_mask(frame_rgb, is_moving)

        # A. Create Seed from Body
        _, seed_mask = cv2.threshold(mp_mask, 0.5, 1.0, cv2.THRESH_BINARY)
        seed_mask = seed_mask.astype(np.uint8)

        # B. Dynamic Search Area
        # If moving, expand the search area massively (51x51 kernel)
        # because the hand might be far from where MediaPipe thinks the body is.
        if is_moving:
            search_area = cv2.dilate(seed_mask, self.kernel_search_moving, iterations=3)
        else:
            search_area = cv2.dilate(seed_mask, self.kernel_search_static, iterations=3)

        # C. Filter Skin
        valid_skin = cv2.bitwise_and(skin_mask, skin_mask, mask=search_area)

        # D. Combine
        combined_mask = np.maximum(mp_mask, valid_skin)

        # 4. Temporal Smoothing
        current_alpha = self.base_alpha
        # If moving fast, disable smoothing to prevent "ghosting" lag
        if self.motion_score > 0.05:
            current_alpha = 0.0

        if self.prev_mask is not None:
            final_mask = (combined_mask * (1.0 - current_alpha)) + (self.prev_mask * current_alpha)
        else:
            final_mask = combined_mask

        # 5. Output
        _, output_binary = cv2.threshold(final_mask, 0.5, 1.0, cv2.THRESH_BINARY)

        self.prev_gray = frame_gray
        self.prev_mask = final_mask

        return np.expand_dims(output_binary, axis=-1)

    def get_motion_score(self):
        return self.motion_score