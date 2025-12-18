from typing import List, Dict, Any

class SmoothedLM:
    """
    A simple data container for smoothed 3D coordinates.
    Acts as a drop-in replacement for the immutable MediaPipe landmark objects.
    """
    def __init__(self, x: float, y: float, z: float) -> None: 
        self.x = x
        self.y = y
        self.z = z

class LandmarkSmoother:
    """
    Applies a Low-Pass Filter (Exponential Moving Average) to landmark coordinates.
    
    This reduces the 'jitter' or high-frequency noise common in computer vision 
    tracking, making the cursor or hand movement appear smoother and more organic.
    """

    def __init__(self, alpha: float = 0.6) -> None:
        """
        Initialize the smoother configuration.

        Args:
            alpha (float): The smoothing factor (0.0 to 1.0).
                           - Higher alpha (e.g., 0.9): More responsive, less smooth (trusts new data).
                           - Lower alpha (e.g., 0.1): Very smooth, high latency (trusts history).
                           - Default 0.6 is a balanced starting point.
        """
        self.alpha: float = alpha
        
        # Cache to store the previous frame's smoothed landmarks for each hand (Left/Right)
        self.prev_landmarks: Dict[str, List[Any]] = {}

    def smooth(self, current_landmarks: List[Any], hand_label: str) -> List[Any]:
        """
        Applies the smoothing formula to the current frame's landmarks.

        Args:
            current_landmarks (List[Any]): List of raw landmarks from the tracker.
            hand_label (str): Identifier for the hand ('Left' or 'Right') to maintain separate histories.

        Returns:
            List[Any]: A list of SmoothedLM objects containing the filtered coordinates.
        """
        # 1. Initialization Check
        # If this is the first time we see this hand, we have no history to smooth against.
        # So, we just return the raw data and store it as the 'previous' state.
        if hand_label not in self.prev_landmarks:
            self.prev_landmarks[hand_label] = current_landmarks
            return current_landmarks
        
        smoothed_hand: List[SmoothedLM] = []
        prev_hand = self.prev_landmarks[hand_label]

        # 2. Apply Filter Point-by-Point
        # We iterate through all 21 hand landmarks
        for i, curr_lm in enumerate(current_landmarks):
            prev_lm = prev_hand[i]

            # Formula: Exponential Moving Average (EMA)
            # New_Val = (Current_Raw * alpha) + (Previous_Smoothed * (1 - alpha))
            new_x = (curr_lm.x * self.alpha) + (prev_lm.x * (1 - self.alpha))
            new_y = (curr_lm.y * self.alpha) + (prev_lm.y * (1 - self.alpha))
            new_z = (curr_lm.z * self.alpha) + (prev_lm.z * (1 - self.alpha))

            smoothed_hand.append(SmoothedLM(new_x, new_y, new_z))
            
        # 3. Update History
        # Save the calculated smoothed values to be used as 'prev_hand' in the next frame
        self.prev_landmarks[hand_label] = smoothed_hand
        
        return smoothed_hand