import cv2
import numpy as np


def apply_glitch(frame_rgb, mask, shift=20):
    # Create the Glitched Background
    r, g, b = cv2.split(frame_rgb)

    r_shifted = np.roll(r, -shift, axis=1)
    b_shifted = np.roll(b, shift, axis=1)

    glitched_bg = cv2.merge([r_shifted, g, b_shifted])

    mask_3c = np.repeat(mask, 3, axis=2)
    out = (frame_rgb * mask_3c) + (glitched_bg * (1.0 - mask_3c))

    return out.astype(np.uint8)



def apply_pixelation(frame_rgb, mask, blocks=20):
    # Pixelates the background.
    h, w = frame_rgb.shape[:2]

    small = cv2.resize(frame_rgb, (w // blocks, h // blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_3c = np.repeat(mask, 3, axis=2)
    out = (frame_rgb * mask_3c) + (pixelated * (1.0 - mask_3c))

    return out.astype(np.uint8)