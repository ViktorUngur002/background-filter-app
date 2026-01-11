import cv2
import numpy as np

class SmartCameraman:
    def __init__(self):
        self.current_rect = None
        self.alpha = 0.1

    def process(self, frame, mask):
        h_img, w_img = frame.shape[:2]

        if mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask

        if len(mask_uint8.shape) == 3:
            if mask_uint8.shape[2] == 1:
                mask_uint8 = mask_uint8.squeeze(axis=2)
            else:
                mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return frame

        largest_c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_c)

        target_h = h
        target_y = y
        pad_y_top = int(h * 0.3)
        pad_y_bot = int(h * 0.3)

        pad_x = int(w * 0.4)

        final_x = max(0, x - pad_x)
        final_y = max(0, target_y - pad_y_top)
        final_w = min(w_img - final_x, w + 2 * pad_x)
        final_h = min(h_img - final_y, target_h + pad_y_top + pad_y_bot)

        target_aspect = w_img / h_img
        current_aspect = final_w / final_h

        if current_aspect > target_aspect:
            new_h = int(final_w / target_aspect)

            diff = new_h - final_h
            final_y = max(0, final_y - diff // 2)
            final_h = min(h_img - final_y, new_h)
        else:
            new_w = int(final_h * target_aspect)
            diff = new_w - final_w
            final_x = max(0, final_x - diff // 2)
            final_w = min(w_img - final_x, new_w)

        target_rect = np.array([final_x, final_y, final_w, final_h], dtype=np.float32)

        if self.current_rect is None:
            self.current_rect = target_rect
        else:
            self.current_rect = (self.current_rect * (1 - self.alpha)) + (target_rect * self.alpha)

        cx, cy, cw, ch = self.current_rect.astype(int)

        cx = max(0, cx)
        cy = max(0, cy)

        cw = min(w_img - cx, cw)
        ch = min(h_img - cy, ch)

        cw = max(1, cw)
        ch = max(1, ch)

        cropped = frame[cy:cy + ch, cx:cx + cw]

        if cropped.size == 0:
            return frame

        output = cv2.resize(cropped, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

        return output