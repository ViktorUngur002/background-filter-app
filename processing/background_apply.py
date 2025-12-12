import cv2
import numpy as np


def apply_blur_background(frame_rgb, mask, motion_score=0.0):
    # --- DOCUMENTATION ALIGNMENT: Adaptive Gaussian Kernel ---
    # The blur strength adapts based on the motion score.
    # Less motion = softer blur (e.g., 21).
    # High motion = heavy blur (e.g., 61) to hide background distractions during movement.

    # Base blur amount
    base_k = 21
    # Adaptive factor: Scale motion_score (0.0-0.1 usually) to kernel size
    adaptive_k = int(motion_score * 500)

    # Ensure kernel size is odd and positive
    total_k = base_k + adaptive_k
    if total_k % 2 == 0:
        total_k += 1

    # Apply the Adaptive Gaussian Kernel
    blurred_bg = cv2.GaussianBlur(frame_rgb, (total_k, total_k), 0)

    mask_3c = np.repeat(mask, 3, axis=2)

    # Blend
    out = frame_rgb * mask_3c + blurred_bg * (1.0 - mask_3c)
    return out.astype(np.uint8)


def apply_pattern_background(frame_rgb, mask, pattern_img):
    if isinstance(pattern_img, np.ndarray):
        bg = pattern_img
    else:
        bg = np.array(pattern_img)  # Fixed typo from ndarray to array

    bg = cv2.resize(bg, (frame_rgb.shape[1], frame_rgb.shape[0]))
    mask_3c = np.repeat(mask, 3, axis=2)
    out = frame_rgb * mask_3c + bg * (1.0 - mask_3c)
    return out.astype(np.uint8)