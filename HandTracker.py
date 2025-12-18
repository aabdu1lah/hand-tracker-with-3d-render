import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional

# Default path to the model bundle
MODEL_PATH: str = 'hand_landmarker.task'

class HandTracker:
    """
    A wrapper for MediaPipe's Hand Landmarker solution.
    
    This class handles the initialization of the ML graph and manages the 
    asynchronous processing loop required for live video streams.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        """
        Initialize the MediaPipe HandLandmarker in LIVE_STREAM mode.

        Args:
            model_path (str): Path to the .task model file.
        """
        # Holder for the most recent inference result
        self.result: Optional[vision.HandLandmarkerResult] = None
        
        # Track timestamps to ensure we only feed strictly increasing times to the graph
        self.latest_timestamp_ms: int = 0

        # 1. Load Model options
        base_options = python.BaseOptions(model_asset_path=model_path)

        # 2. Configure the Landmarker
        # LIVE_STREAM mode is asynchronous; it returns results via the 'result_callback'
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,                       # Track up to 2 hands
            min_hand_detection_confidence=0.3, # Lower threshold = faster but more false positives
            min_tracking_confidence=0.3,       # Lower threshold = less jitter, but might lose tracking
            result_callback=self.print_result  # The function called when AI finishes a frame
        )

        # 3. Create the Landmarker instance
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def print_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        """
        Callback function invoked by MediaPipe when a frame is processed.
        
        Note:
            This runs on a separate thread managed by MediaPipe. We simply 
            update the shared 'self.result' state.
        
        Args:
            result: The detection result containing landmarks and handedness.
            output_image: The image processed (unused here, but required by signature).
            timestamp_ms: The timestamp of the processed frame.
        """
        self.result = result

    def detect_async(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """
        Sends a frame to the MediaPipe graph for processing.

        Args:
            frame (np.ndarray): The raw image frame (usually from OpenCV, BGR).
            timestamp_ms (int): The current timestamp in milliseconds.
        
        Note:
            MediaPipe requires strictly monotonically increasing timestamps. 
            If a frame arrives out of order or with a duplicate timestamp, it is dropped.
        """
        # Convert OpenCV image (numpy) to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Ensure strict monotonicity of timestamps
        if timestamp_ms > self.latest_timestamp_ms:
            self.landmarker.detect_async(mp_image, timestamp_ms)
            self.latest_timestamp_ms = timestamp_ms

    def get_latest_result(self) -> Optional[vision.HandLandmarkerResult]:
        """
        Retrieves the most recent inference result.

        Returns:
            Optional[vision.HandLandmarkerResult]: The latest result or None if no data yet.
        """
        return self.result