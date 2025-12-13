import cv2
import numpy as np


class SmartCameraman:
    def __init__(self):
        self.current_rect = None
        # Smoothing Factor: Lower = smoother.
        self.alpha = 0.1

    def process(self, frame, mask):
        h_img, w_img = frame.shape[:2]

        # 1. Mask Preparation
        if mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask

        if len(mask_uint8.shape) == 3:
            if mask_uint8.shape[2] == 1:
                mask_uint8 = mask_uint8.squeeze(axis=2)
            else:
                mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)

        # 2. Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return frame

            # 3. Find target bounding box
        largest_c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_c)

        # --- FIX: TOP MAGNET LOGIC ---
        # When close, pixels are larger. We relax the threshold from 10 to 60.
        # This catches "almost touching" cases.
        is_near_top = y < 60

        # If near top, we assume the head is cut off.
        # We assume height is at least 1.3x width (standard human torso ratio).
        implied_height = int(w * 1.3)

        if is_near_top:
            # 1. Expand Height: Use the larger of real height or implied height
            target_h = max(h, implied_height)

            # 2. Lock Y to 0: Force the camera to look at the absolute top edge
            target_y = 0

            # 3. Padding: Reduce top padding to 0 so we don't push the view down
            pad_y_top = 0
            pad_y_bot = int(h * 0.3)
        else:
            # Standard tracking
            target_h = h
            target_y = y
            pad_y_top = int(h * 0.3)
            pad_y_bot = int(h * 0.3)

        # 4. Apply Padding
        pad_x = int(w * 0.4)

        final_x = max(0, x - pad_x)
        final_y = max(0, target_y - pad_y_top)
        final_w = min(w_img - final_x, w + 2 * pad_x)
        # Use target_h (which might include the "ghost" head)
        final_h = min(h_img - final_y, target_h + pad_y_top + pad_y_bot)

        # 5. Aspect Ratio Correction
        # We must adjust dimensions to match the screen aspect ratio (e.g. 16:9)
        target_aspect = w_img / h_img
        current_aspect = final_w / final_h

        if current_aspect > target_aspect:
            # Crop is too wide -> Increase height
            new_h = int(final_w / target_aspect)

            if is_near_top:
                # CRITICAL FIX: If near top, expand DOWNWARDS only.
                # Don't try to center the expansion, or it will push Y negative (invalid)
                final_h = min(h_img - final_y, new_h)
            else:
                # Standard: Expand center
                diff = new_h - final_h
                final_y = max(0, final_y - diff // 2)
                final_h = min(h_img - final_y, new_h)

        else:
            # Crop is too tall -> Increase width
            new_w = int(final_h * target_aspect)
            diff = new_w - final_w
            final_x = max(0, final_x - diff // 2)
            final_w = min(w_img - final_x, new_w)

        # 6. Smooth Transition
        target_rect = np.array([final_x, final_y, final_w, final_h], dtype=np.float32)

        if self.current_rect is None:
            self.current_rect = target_rect
        else:
            self.current_rect = (self.current_rect * (1 - self.alpha)) + (target_rect * self.alpha)

        # 7. Crop and Resize
        cx, cy, cw, ch = self.current_rect.astype(int)

        # Safety bounds checks
        cx = max(0, cx);
        cy = max(0, cy)

        # Prevent going out of bounds
        cw = min(w_img - cx, cw)
        ch = min(h_img - cy, ch)

        # Prevent 0-size crash
        cw = max(1, cw);
        ch = max(1, ch)

        cropped = frame[cy:cy + ch, cx:cx + cw]

        if cropped.size == 0:
            return frame

        output = cv2.resize(cropped, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

        return output