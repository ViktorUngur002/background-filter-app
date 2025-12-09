import mediapipe as mp
import cv2
import numpy as np


class PersonSegmenter:
    def __init__(self):
        # MediaPipe selfie segmentation
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

        # Temporal smoothing memory
        self.prev_mask = None
        self.prev_frame = None
        self.current_alpha = 0.3  # Start with faster transitions

        # Pre-create kernels for morphological operations (optimization)
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_finger = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # For finger gaps
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def get_mask(self, frame_rgb):
        """Main entry point - returns refined segmentation mask"""
        # Get base mask from MediaPipe
        result = self.segmenter.process(frame_rgb)
        base_mask = result.segmentation_mask

        if len(base_mask.shape) == 3:
            base_mask = base_mask.squeeze()

        # Run optimized refinement pipeline
        refined_mask = self.refine_mask(base_mask, frame_rgb)

        if len(refined_mask.shape) == 2:
            refined_mask = refined_mask[:, :, None]

        return refined_mask

    def refine_mask(self, mask, frame_rgb):
        """Optimized multi-stage mask refinement pipeline"""

        # --- STAGE 1: FAST MOTION-ADAPTIVE TEMPORAL SMOOTHING ---
        mask = self.apply_fast_temporal_smoothing(mask, frame_rgb)

        # --- STAGE 2: AGGRESSIVE THRESHOLDING (Lower threshold for faster response) ---
        mask = self.apply_fast_threshold(mask)

        # --- STAGE 3: FINGER GAP DETECTION & MORPHOLOGICAL REFINEMENT ---
        mask = self.apply_finger_gap_detection(mask, frame_rgb)

        # --- STAGE 4: FAST EDGE-PRESERVING SMOOTHING ---
        mask = self.apply_fast_bilateral_smoothing(mask)

        # --- STAGE 5: EDGE SHARPENING ---
        mask = self.apply_fast_edge_sharpening(mask, frame_rgb)

        return mask

    def apply_fast_temporal_smoothing(self, mask, frame_rgb):
        """Optimized temporal smoothing with faster transitions"""
        if self.prev_mask is None or self.prev_frame is None:
            self.prev_mask = mask
            self.prev_frame = frame_rgb.copy()
            return mask

        # Downsample for faster motion calculation
        small_curr = cv2.resize(frame_rgb, (160, 90), interpolation=cv2.INTER_LINEAR)
        small_prev = cv2.resize(self.prev_frame, (160, 90), interpolation=cv2.INTER_LINEAR)

        # Calculate motion score (much faster on smaller image)
        gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_RGB2GRAY)
        frame_diff = cv2.absdiff(gray_curr, gray_prev)
        motion_score = np.mean(frame_diff) / 255.0

        # More aggressive alpha adaptation for faster transitions
        if motion_score > 0.06:
            target_alpha = 0.05  # Very fast update for motion
        elif motion_score > 0.03:
            target_alpha = 0.2  # Medium-fast
        else:
            target_alpha = 0.5  # Still relatively fast when static

        # Faster alpha transition
        self.current_alpha = 0.5 * self.current_alpha + 0.5 * target_alpha

        # Apply temporal smoothing
        smoothed_mask = (mask * (1.0 - self.current_alpha)) + (self.prev_mask * self.current_alpha)

        # Update history
        self.prev_mask = smoothed_mask.copy()
        self.prev_frame = frame_rgb.copy()

        return smoothed_mask

    def apply_fast_threshold(self, mask):
        """Fast aggressive thresholding for quicker response"""
        # Lower threshold = faster to include new areas as foreground
        _, binary_mask = cv2.threshold(mask, 0.28, 1.0, cv2.THRESH_BINARY)
        return binary_mask

    def apply_finger_gap_detection(self, mask, frame_rgb):
        """Advanced finger gap detection using edge-aware morphology"""
        mask_uint8 = (mask * 255).astype(np.uint8)

        # STEP 1: Detect image edges (to find finger boundaries)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Use simpler edge detection (faster than Canny)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

        # Calculate gradient magnitude
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # Threshold to get strong edges only
        _, strong_edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

        # STEP 2: Erode mask slightly to separate touching fingers
        # This creates gaps between fingers that are close together
        eroded = cv2.erode(mask_uint8, self.kernel_finger, iterations=1)

        # STEP 3: Find the mask boundary (where foreground meets background)
        dilated = cv2.dilate(mask_uint8, self.kernel_small, iterations=1)
        boundary = cv2.subtract(dilated, mask_uint8)

        # STEP 4: Where we have strong edges AND mask boundary, likely a finger gap
        # This identifies the space between fingers
        finger_gaps = cv2.bitwise_and(strong_edges, boundary)

        # Dilate finger gaps slightly to ensure they're captured
        finger_gaps = cv2.dilate(finger_gaps, self.kernel_finger, iterations=1)

        # STEP 5: Subtract finger gaps from mask (mark them as background)
        result = cv2.subtract(eroded, finger_gaps)

        # STEP 6: Clean up noise while preserving finger gaps
        # Opening removes small noise
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel_small)

        # STEP 7: Fill small holes in body/clothing (but not finger gaps)
        # Use larger kernel for closing to fill body holes but preserve finger gaps
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, self.kernel_medium)

        # STEP 8: Final noise reduction with median filter
        result = cv2.medianBlur(result, 5)

        return result.astype(np.float32) / 255.0

    def apply_fast_bilateral_smoothing(self, mask):
        """Fast bilateral filter for edge-preserving smoothing"""
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Reduced parameters for speed: smaller diameter and sigma values
        smoothed = cv2.bilateralFilter(mask_uint8, d=5, sigmaColor=30, sigmaSpace=30)

        return smoothed.astype(np.float32) / 255.0

    def apply_fast_edge_sharpening(self, mask, frame_rgb):
        """Lightweight edge sharpening using unsharp masking"""
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Fast unsharp mask: original - blurred
        blurred = cv2.GaussianBlur(mask_uint8, (3, 3), 0)
        sharpened = cv2.addWeighted(mask_uint8, 1.4, blurred, -0.4, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # One more light bilateral for final smoothing
        result = cv2.bilateralFilter(sharpened, d=3, sigmaColor=20, sigmaSpace=20)

        return result.astype(np.float32) / 255.0