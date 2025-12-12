import mediapipe as mp
import cv2
import numpy as np


class PersonSegmenter:
    def __init__(self):
        # MediaPipe selfie segmentation
        self.mp_selfie = mp.solutions.selfie_segmentation
        # creates an instance of the MediaPipe SelfieSegmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

        # Temporal smoothing memory
        # stores the previous frame's segmentation mask, used to blend current and past masks for stability
        self.prev_mask = None
        # stores the previous RGB frame to compute motion between frames
        self.prev_frame = None
        # temporal smoothing amount
        self.current_alpha = 0.3

        # Pre-create kernels for morphological operations
        self.kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_finger = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

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
        """Optimized multi-stage mask refinement pipeline with skin detection"""

        # --- STAGE 1: DETECT SKIN REGIONS ---
        skin_mask = self.detect_skin_regions(frame_rgb)

        # --- STAGE 2: FAST MOTION-ADAPTIVE TEMPORAL SMOOTHING ---
        mask = self.apply_fast_temporal_smoothing(mask, frame_rgb)

        # --- STAGE 3: SKIN-AWARE THRESHOLDING ---
        mask = self.apply_skin_aware_threshold(mask, skin_mask)

        # --- STAGE 4: ENHANCE HAND REGIONS ---
        mask = self.enhance_hand_regions(mask, skin_mask, frame_rgb)

        # --- STAGE 5: FINGER GAP DETECTION & MORPHOLOGICAL REFINEMENT ---
        mask = self.apply_finger_gap_detection(mask, frame_rgb)

        # --- STAGE 6: FAST EDGE-PRESERVING SMOOTHING ---
        mask = self.apply_fast_bilateral_smoothing(mask)

        # --- STAGE 7: EDGE SHARPENING WITH SKIN GUIDANCE ---
        mask = self.apply_guided_edge_sharpening(mask, frame_rgb, skin_mask)

        return mask

    def detect_skin_regions(self, frame_rgb):
        """Detect skin regions using YCrCb color space - highly effective for hands/face"""
        # Convert to YCrCb color space (better for skin detection than RGB)
        ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)

        # Skin color range in YCrCb (covers most skin tones)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Additional HSV-based skin detection for better coverage
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Combine both masks
        skin_mask = cv2.bitwise_or(skin_mask, skin_mask_hsv)

        # Remove small noise
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel_small)

        # Dilate slightly to ensure we capture hand edges
        skin_mask = cv2.dilate(skin_mask, self.kernel_medium, iterations=2)

        # Apply Gaussian blur for smooth transitions
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        return skin_mask.astype(np.float32) / 255.0

    def apply_fast_temporal_smoothing(self, mask, frame_rgb):
        """Optimized temporal smoothing with faster transitions"""
        if self.prev_mask is None or self.prev_frame is None:
            self.prev_mask = mask
            self.prev_frame = frame_rgb.copy()
            return mask

        # Downsample for faster motion calculation
        small_curr = cv2.resize(frame_rgb, (160, 90), interpolation=cv2.INTER_LINEAR)
        small_prev = cv2.resize(self.prev_frame, (160, 90), interpolation=cv2.INTER_LINEAR)

        # Calculate motion score
        gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_RGB2GRAY)
        frame_diff = cv2.absdiff(gray_curr, gray_prev)
        motion_score = np.mean(frame_diff) / 255.0

        # Aggressive alpha adaptation for faster transitions
        if motion_score > 0.06:
            target_alpha = 0.05
        elif motion_score > 0.03:
            target_alpha = 0.2
        else:
            target_alpha = 0.5

        self.current_alpha = 0.5 * self.current_alpha + 0.5 * target_alpha

        # Apply temporal smoothing
        smoothed_mask = (mask * (1.0 - self.current_alpha)) + (self.prev_mask * self.current_alpha)

        # Update history
        self.prev_mask = smoothed_mask.copy()
        self.prev_frame = frame_rgb.copy()

        return smoothed_mask

    def apply_skin_aware_threshold(self, mask, skin_mask):
        """Adaptive thresholding that's more aggressive in skin regions"""
        # Lower threshold in skin regions (easier to include hands)
        # Higher threshold in non-skin regions (more conservative)
        threshold_map = np.where(skin_mask > 0.3, 0.20, 0.30)

        # Apply adaptive threshold
        binary_mask = (mask > threshold_map).astype(np.float32)

        return binary_mask

    def enhance_hand_regions(self, mask, skin_mask, frame_rgb):
        """Special processing to enhance hand/skin regions in the mask"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        skin_uint8 = (skin_mask * 255).astype(np.uint8)

        # Find regions that are skin but not in mask (potential hand areas being cut off)
        potential_hand_regions = cv2.bitwise_and(skin_uint8, cv2.bitwise_not(mask_uint8))

        # Only consider regions that are connected to the existing mask (not random skin-colored objects)
        # Dilate mask to find nearby regions
        dilated_mask = cv2.dilate(mask_uint8, self.kernel_large, iterations=2)

        # Get potential hand regions that touch the mask
        connected_hand_regions = cv2.bitwise_and(potential_hand_regions, dilated_mask)

        # Add these regions to the mask
        enhanced_mask = cv2.add(mask_uint8, connected_hand_regions)

        # Also enforce that strong skin regions within the mask boundary stay in the mask
        # This prevents the background from bleeding through the hand
        hand_core = cv2.bitwise_and(skin_uint8, mask_uint8)
        _, strong_hand = cv2.threshold(hand_core, 100, 255, cv2.THRESH_BINARY)

        # Ensure strong hand regions are definitely included
        enhanced_mask = cv2.bitwise_or(enhanced_mask, strong_hand)

        return enhanced_mask.astype(np.float32) / 255.0

    def apply_finger_gap_detection(self, mask, frame_rgb):
        """Advanced finger gap detection using edge-aware morphology"""
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Detect image edges
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Use Sobel for edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # Get strong edges
        _, strong_edges = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)

        # Erode mask to separate fingers
        eroded = cv2.erode(mask_uint8, self.kernel_finger, iterations=1)

        # Find mask boundary
        dilated = cv2.dilate(mask_uint8, self.kernel_small, iterations=1)
        boundary = cv2.subtract(dilated, mask_uint8)

        # Identify finger gaps
        finger_gaps = cv2.bitwise_and(strong_edges, boundary)
        finger_gaps = cv2.dilate(finger_gaps, self.kernel_tiny, iterations=1)

        # Remove finger gaps from mask
        result = cv2.subtract(eroded, finger_gaps)

        # Clean up with morphological operations
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel_small)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, self.kernel_medium)

        # Median filter for final cleanup
        result = cv2.medianBlur(result, 5)

        return result.astype(np.float32) / 255.0

    def apply_fast_bilateral_smoothing(self, mask):
        """Fast bilateral filter for edge-preserving smoothing"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(mask_uint8, d=5, sigmaColor=30, sigmaSpace=30)
        return smoothed.astype(np.float32) / 255.0

    def apply_guided_edge_sharpening(self, mask, frame_rgb, skin_mask):
        """Edge sharpening with special attention to hand edges"""
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Detect mask edges
        mask_edges = cv2.Canny(mask_uint8, 50, 150)

        # Detect image edges
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.Canny(gray, 30, 100)

        # Find edges that exist in both mask and image (true object boundaries)
        true_edges = cv2.bitwise_and(mask_edges, img_edges)

        # Dilate to create edge region
        edge_region = cv2.dilate(true_edges, self.kernel_small, iterations=2)
        edge_region = edge_region.astype(np.float32) / 255.0

        # Apply stronger sharpening at true edges
        blurred = cv2.GaussianBlur(mask_uint8, (3, 3), 0)

        # More aggressive sharpening in skin regions (hands need sharper edges)
        skin_uint8 = (skin_mask * 255).astype(np.uint8)
        skin_edges = cv2.bitwise_and(edge_region.astype(np.uint8), skin_uint8)

        # Strong sharpening for hand edges
        sharp_factor = np.where(skin_edges > 0, 1.6, 1.4)
        blur_factor = np.where(skin_edges > 0, -0.6, -0.4)

        sharpened = mask_uint8.astype(np.float32) * sharp_factor + blurred.astype(np.float32) * blur_factor
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Apply edge sharpening only at detected edges
        result = mask_uint8 * (1 - edge_region) + sharpened * edge_region
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Final light bilateral filter
        result = cv2.bilateralFilter(result, d=3, sigmaColor=20, sigmaSpace=20)

        return result.astype(np.float32) / 255.0
