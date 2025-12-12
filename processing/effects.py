import cv2
import numpy as np


def apply_glitch(frame_rgb, mask, shift=20):
    """
    Splits RGB channels and shifts them to create a Chromatic Aberration (Glitch) effect
    on the background, keeping the user normal.
    """
    # 1. Create the Glitched Background
    r, g, b = cv2.split(frame_rgb)

    # Shift Red channel to the left
    r_shifted = np.roll(r, -shift, axis=1)
    # Shift Blue channel to the right
    b_shifted = np.roll(b, shift, axis=1)
    # Green stays still

    glitched_bg = cv2.merge([r_shifted, g, b_shifted])

    # 2. Blend: User (Normal) + Background (Glitched)
    mask_3c = np.repeat(mask, 3, axis=2)

    # Note: mask is float (0.0-1.0), frame is uint8.
    # We perform math in float, then cast back.
    out = (frame_rgb * mask_3c) + (glitched_bg * (1.0 - mask_3c))

    return out.astype(np.uint8)



def apply_pixelation(frame_rgb, mask, blocks=20):
    """
    Pixelates the background.
    """
    h, w = frame_rgb.shape[:2]

    # 1. Downscale
    small = cv2.resize(frame_rgb, (w // blocks, h // blocks), interpolation=cv2.INTER_LINEAR)

    # 2. Upscale (Nearest Neighbor creates the blocks)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. Blend
    mask_3c = np.repeat(mask, 3, axis=2)
    out = (frame_rgb * mask_3c) + (pixelated * (1.0 - mask_3c))

    return out.astype(np.uint8)