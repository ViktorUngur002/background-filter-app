import mediapipe as mp
import numpy as np
import cv2

class PersonSegmenter:
    def __init__(self):
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def get_mask(self, frame_rgb):
        result = self.segmenter.process(frame_rgb)
        mask = result.segmentation_mask

        if len(mask.shape) == 2:
            mask = mask[:, :, None]

        return mask