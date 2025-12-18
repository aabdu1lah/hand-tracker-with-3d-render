from typing import Dict, List, Any

class GestureEngine:
    """
    Analyzes hand geometry landmarks to detect specific intent, such as 'Pinch' gestures.
    
    This engine implements Hysteresis (Schmitt Trigger logic) to ensure stable 
    state transitions. This prevents the system from flickering rapidly between 
    'Pinch' and 'Open' states when the user's hand hovers right at the threshold boundary.
    """

    def __init__(self, pinch_threshold: float = 0.05, release_threshold: float = 0.07) -> None:
        """
        Initialize the gesture engine with specific sensitivity thresholds.

        Args:
            pinch_threshold (float): The normalized distance (0.0-1.0) below which a pinch 
                                     is registered. Lower = harder to pinch.
            release_threshold (float): The normalized distance above which a pinch 
                                       is released. Must be > pinch_threshold to create a buffer.
        """
        self.pinch_thresh: float = pinch_threshold
        self.release_thresh: float = release_threshold
        
        # Tracks the current state (True=Pinching, False=Open) for each hand (Left/Right)
        # We store this to implement stateful logic (Hysteresis)
        self.state: Dict[str, bool] = {'Left': False, 'Right': False}

    def detect_pinch(self, landmarks: List[Any], label: str) -> bool:
        """
        Calculates the distance between the Thumb and Index finger to determine pinch state.

        Args:
            landmarks (List[Any]): A list of 21 normalized landmark objects (must have .x, .y, .z attributes).
                                   Index 4 is Thumb Tip, Index 8 is Index Finger Tip.
            label (str): The label of the hand being processed ('Left' or 'Right').

        Returns:
            bool: True if the hand is currently in a 'Pinch' state, False otherwise.
        """
        # 1. Extract Key Landmarks
        # MediaPipe Hand landmarks: 4 = Thumb Tip, 8 = Index Finger Tip
        thumb = landmarks[4]
        index = landmarks[8]

        # 2. Calculate 3D Euclidean Distance
        # Formula: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
        # Note: Coordinates are normalized (0.0 to 1.0). 
        # A distance of 0.05 is roughly 5% of the total screen dimension.
        dist: float = (
            (thumb.x - index.x)**2 + 
            (thumb.y - index.y)**2 + 
            (thumb.z - index.z)**2
        )**0.5

        # 3. Hysteresis Logic (State Machine)
        # We check the CURRENT state to decide which threshold to use.
        if self.state.get(label, False):
            # CASE: Currently Pinching (True)
            # We only release the pinch if the fingers move FAR apart (passed release_thresh).
            # This 'stickiness' prevents the pinch from dropping if the user slightly relaxes their hand.
            if dist > self.release_thresh:
                self.state[label] = False
        else:
            # CASE: Currently Open (False)
            # We only trigger a pinch if fingers get VERY close (passed pinch_thresh).
            if dist < self.pinch_thresh:
                self.state[label] = True

        return self.state[label]