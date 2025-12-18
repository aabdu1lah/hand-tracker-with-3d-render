from typing import Tuple

class CoordinateMapper:
    """
    Handles the transformation of raw normalized 2D image coordinates (0.0 to 1.0)
    into a centered 3D Cartesian coordinate system suitable for interaction.

    This class performs three main functions:
    1. Centering: Moves the origin from top-left (0,0) to the center of the frame.
    2. Aspect Ratio Correction: Ensures X movements match Y movements 1:1.
    3. Mirroring: Flips the X-axis so the interaction feels like a mirror.
    """

    def __init__(self, width: int, height: int, z_close: float = 4.0, z_far: float = 10.0) -> None:
        """
        Initialize the mapper with screen dimensions and depth bounds.

        Args:
            width (int): The width of the input frame in pixels.
            height (int): The height of the input frame in pixels.
            z_close (float, optional): The 'near' clipping plane for depth inputs. Defaults to 4.0.
            z_far (float, optional): The 'far' clipping plane for depth inputs. Defaults to 10.0.
        """
        self.width: int = width
        self.height: int = height
        
        # Calculate aspect ratio (e.g., 16/9 ≈ 1.77) to scale X coordinates
        # so that 1 unit of movement in X equals 1 unit of movement in Y.
        self.aspect_ratio: float = width / height
        
        self.z_min: float = z_close
        self.z_max: float = z_far

    def normalize(self, x_raw: float, y_raw: float, z_raw: float) -> Tuple[float, float, float]:
        """
        Transforms raw landmark coordinates into normalized world space centered at (0,0).

        Args:
            x_raw (float): Normalized X from tracker (0.0 left -> 1.0 right).
            y_raw (float): Normalized Y from tracker (0.0 top -> 1.0 bottom).
            z_raw (float): Raw depth estimate (usually inverse size or distance).

        Returns:
            Tuple[float, float, float]: (x, y, z) coordinates in a centered 3D space.
                - x: Centered, aspect-corrected, and mirrored.
                - y: Centered and inverted (Up is positive).
                - z: Normalized between -1.0 and 1.0.
        """
        # --- X AXIS TRANSFORMATION ---
        # 1. (x_raw - 0.5): Centers the data so 0.5 becomes 0.0.
        # 2. * self.aspect_ratio: Expands X range to preserve squareness (avoids squeezing).
        # 3. Negative sign (-): Performs the "Mirror" flip. 
        #    Without this, moving your hand right would move the cursor left (camera perspective).
        #    With this, moving your hand right moves the cursor right (mirror perspective).
        x: float = - (x_raw - 0.5) * self.aspect_ratio

        # --- Z AXIS (DEPTH) TRANSFORMATION ---
        # Clamp the raw Z value to defined bounds to prevent extreme spikes
        z_clamped: float = max(self.z_min, min(z_raw, self.z_max))
        
        # Normalize Z to a 0.0 - 1.0 range based on the min/max calibration
        z_norm: float = (z_clamped - self.z_min) / (self.z_max - self.z_min)
        
        # Remap 0..1 to -1..1 range (standard for many 3D engines)
        z: float = z_norm * 2 - 1

        # --- Y AXIS TRANSFORMATION ---
        # 1. (y_raw - 0.5): Centers the data.
        # 2. Negative sign (-): Inverts the axis.
        #    Computer Vision uses (0,0) at Top-Left (Y increases downwards).
        #    3D World Space uses Y increasing upwards. This corrects that orientation.
        y: float = - (y_raw - 0.5)

        return x, y, z