import cv2
import os
import numpy as np
from typing import Tuple, Optional

# --- Configuration Constants ---
WEBCAM_ID: int = 0         # The index of the camera (0 is usually the default built-in or USB cam)
WIDTH: int = 1280          # Desired width resolution
HEIGHT: int = 720          # Desired height resolution
TARGET_FPS: int = 30       # Desired frames per second
CODEC: str = 'MJPG'        # Motion JPEG codec (usually offers high frame rates at HD res)


class Capture:
    """
    A wrapper class for cv2.VideoCapture that enforces specific camera settings 
    via Video4Linux2 (v4l2) controls before initialization.
    
    Note:
        This class relies on the 'v4l2-ctl' command line utility, which is 
        specific to Linux environments.
    """

    def __init__(self) -> None:
        """
        Initializes the video capture device with specific hardware flags and OpenCV properties.

        Raises:
            IOError: If the webcam cannot be opened.
        """
        # Enforce driver-level settings (Linux only) before OpenCV grabs the handle
        self.force_camera_settings()

        # Initialize the OpenCV VideoCapture object
        self.cap: cv2.VideoCapture = cv2.VideoCapture(WEBCAM_ID)

        # Apply OpenCV properties to the capture stream
        # Note: CODEC must be set first for some cameras to accept higher resolutions
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*CODEC))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        # Verify the camera opened successfully
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam with ID {WEBCAM_ID}")

    def force_camera_settings(self) -> None:
        """
        Executes shell commands to force specific Video4Linux2 (v4l2) settings.
        
        These settings are critical for consistency:
        1. Disables exposure dynamic framerate (prevents FPS drops in low light).
        2. Sets auto-exposure mode (usually 3 = Aperture Priority Mode).
        """
        # Disable dynamic framerate reduction in low light
        # -d /dev/video0 targets the specific device file
        os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0")
        
        # Set Auto Exposure mode (3 is often 'Auto Mode' or 'Aperture Priority')
        os.system("v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=3")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the next frame from the video stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: 
                - bool: True if frame was grabbed successfully, False otherwise.
                - np.ndarray: The image frame (or None if read failed).
        """
        return self.cap.read()

    def release(self) -> None:
        """
        Safely releases the video capture resource to free up the hardware.
        """
        if self.cap.isOpened():
            self.cap.release()