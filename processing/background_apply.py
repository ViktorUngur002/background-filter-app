import cv2
import numpy as np

def apply_blur_background(frame_rgb, mask, blur_amount=45):
    mask_3c = np.repeat(mask,3,axis=2)
    blurred_bg = cv2.GaussianBlur(frame_rgb,(blur_amount,blur_amount),0)

    out = frame_rgb * mask_3c + blurred_bg * (1.0 - mask_3c)
    return out.astype(np.uint8)

def apply_pattern_background(frame_rgb, mask, pattern_img):
    if isinstance(pattern_img, np.ndarray):
        bg = pattern_img
    else:
        bg = np.ndarray(pattern_img)

    bg = cv2.resize(bg, (frame_rgb.shape[1],frame_rgb.shape[0]))

    mask_3c = np.repeat(mask,3,axis=2)

    out = frame_rgb * mask_3c + bg * (1.0 - mask_3c)
    return out.astype(np.uint8)