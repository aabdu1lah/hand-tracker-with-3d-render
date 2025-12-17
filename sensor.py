import json
import cv2
import mediapipe as mp
import time
import os
import numpy as np
import asyncio
import websockets
import base64
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = 'hand_landmarker.task'
WEBCAM_ID = 0
WIDTH, HEIGHT = 1280, 720  # Start with 720p as requested
TARGET_FPS = 30
CODEC = 'MJPG'  # Use MJPEG to reduce compression artifacts
JPEG_QUALITY = 50 # 0-100 (Higher is better quality)

class CoordinateMapper:
    def __init__(self, width, height, z_close=4.0, z_far=10.0):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        
        # Calibration from your test
        self.z_min = z_close  # Hand closest to camera
        self.z_max = z_far    # Hand furthest away

    def normalize(self, x_raw, y_raw, z_raw):
        # 1. Center X/Y to (0,0) being the middle of the screen
        # Range becomes [-0.5, 0.5]
        x = x_raw - 0.5
        y = y_raw - 0.5

        # 2. Fix Aspect Ratio (Stretch X so movement is 1:1 with Y)
        # If we don't do this, a circular motion looks like an oval
        x = x * self.aspect_ratio

        # 3. Map Z to [-1.0 (close), 1.0 (far)]
        # Formula: (value - min) / (max - min) * 2 - 1
        z_clamped = max(self.z_min, min(z_raw, self.z_max)) # Clamp to prevent glitches
        z_norm = (z_clamped - self.z_min) / (self.z_max - self.z_min)
        z = z_norm * 2 - 1
        
        # Optional: Invert Y? 
        # In 3D graphics (Three.js/OpenGL), Y usually goes UP. 
        # In images, Y goes DOWN. Let's flip Y to match 3D standard.
        y = -y 

        return x, y, z
    
class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        # Alpha: 0.0 = infinite lag (no movement), 1.0 = no smoothing (raw jitter)
        # 0.5 - 0.7 is a good sweet spot for hands.
        self.alpha = alpha
        self.prev_landmarks = {}  # Stores previous state: {'Left': [lm, ...], 'Right': [lm, ...]}

    def smooth(self, current_landmarks, hand_label):
        if hand_label not in self.prev_landmarks:
            self.prev_landmarks[hand_label] = current_landmarks
            return current_landmarks

        smoothed_hand = []
        prev_hand = self.prev_landmarks[hand_label]

        for i, curr_lm in enumerate(current_landmarks):
            prev_lm = prev_hand[i]
            
            # Simple Exponential Smoothing (EMA)
            # New = (Current * alpha) + (Previous * (1 - alpha))
            new_x = (curr_lm.x * self.alpha) + (prev_lm.x * (1 - self.alpha))
            new_y = (curr_lm.y * self.alpha) + (prev_lm.y * (1 - self.alpha))
            new_z = (curr_lm.z * self.alpha) + (prev_lm.z * (1 - self.alpha))
            
            # Reconstruct landmark object (using a simple object or similar structure)
            # We reuse the specific MP class or a simple namespace
            class SmoothedLM:
                def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
            
            smoothed_hand.append(SmoothedLM(new_x, new_y, new_z))

        self.prev_landmarks[hand_label] = smoothed_hand
        return smoothed_hand

class HandTracker:
    def __init__(self, model_path=MODEL_PATH):
        self.result = None
        self.latest_timestamp_ms = 0
        
        # Initialize the Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self.print_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def print_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
        """Callback to store the async result."""
        self.result = result

    def detect_async(self, frame, timestamp_ms):
        """Converts frame and sends to MP for inference."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Timestamp must be monotonically increasing
        if timestamp_ms > self.latest_timestamp_ms:
            self.landmarker.detect_async(mp_image, timestamp_ms)
            self.latest_timestamp_ms = timestamp_ms

    def get_latest_result(self):
        return self.result

class DebugVisualizer:
    """Handles drawing and textual logging on the frame."""
    def draw_radar(self, frame, hands_world_data):
        """Draws a top-down view (XZ plane) in the corner."""
        h, w, _ = frame.shape
        radar_size = 200
        radar_img = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
        
        # Draw Background/Grid
        cv2.rectangle(radar_img, (0, 0), (radar_size, radar_size), (30, 30, 30), -1)
        cv2.line(radar_img, (radar_size//2, 0), (radar_size//2, radar_size), (100, 100, 100), 1) # Center Line
        cv2.putText(radar_img, "TOP VIEW (XZ)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for hand in hands_world_data:
            # Map World X (-1 to 1) -> Radar X (0 to 200)
            # Map World Z (-1 to 1) -> Radar Y (200 to 0) 
            # Note: Z=-1 is close (bottom of radar), Z=1 is far (top of radar)
            
            rx = int((hand['x'] + 1) * 0.5 * radar_size)
            ry = int((hand['z'] + 1) * 0.5 * radar_size) 
            
            # Clamp for safety
            rx = max(0, min(radar_size, rx))
            ry = max(0, min(radar_size, ry))

            color = (0, 0, 255) if "Right" in hand['label'] else (255, 0, 0)
            cv2.circle(radar_img, (rx, ry), 8, color, -1)
            cv2.putText(radar_img, hand['label'][0], (rx-4, ry+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Overlay Radar onto main frame (Bottom Right)
        frame[h-radar_size:h, w-radar_size:w] = radar_img
        return frame

    # Update the main draw function to accept the mapper
    def draw(self, frame, detection_result, fps, smoother, mapper):
        annotated_frame = frame.copy()
        
        # ... (Existing FPS code) ...

        hands_world_data = [] # Store processed data for Radar

        if detection_result and detection_result.hand_landmarks:
            for idx, raw_landmarks in enumerate(detection_result.hand_landmarks):
                # 1. Get Label
                label = "Unknown"
                if len(detection_result.handedness) > idx:
                    label = detection_result.handedness[idx][0].category_name
                
                # 2. Smooth
                landmarks = smoother.smooth(raw_landmarks, label)

                # 3. Calculate Raw Depth (Same as before)
                wrist = landmarks[0]
                idx_mcp = landmarks[5]
                dist_x, dist_y = (wrist.x - idx_mcp.x), (wrist.y - idx_mcp.y)
                pixel_scale = (dist_x**2 + dist_y**2)**0.5
                depth_proxy = 1.0 / (pixel_scale + 0.01)

                # 4. NORMALIZE TO WORLD SPACE
                wx, wy, wz = mapper.normalize(wrist.x, wrist.y, depth_proxy)

                # Store for Radar
                hands_world_data.append({'label': label, 'x': wx, 'y': wy, 'z': wz})

                # Draw standard tracking dots on main image
                h, w, _ = frame.shape
                color = (0, 0, 255) if "Right" in label else (255, 0, 0)
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(annotated_frame, (cx, cy), 6, color, -1)
                
                # Display World Coordinates
                text = f"X:{wx:.2f} Y:{wy:.2f} Z:{wz:.2f}"
                cv2.putText(annotated_frame, text, (cx, cy - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw the Radar
        annotated_frame = self.draw_radar(annotated_frame, hands_world_data)
        
        return annotated_frame
    
def force_camera_settings():
    # Disable dynamic framerate to lock 30 FPS
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0")
    # Optional: Ensure auto-exposure is on (since you have fps now, let it handle brightness)
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=3")

CONNECTED_CLIENTS = set()

async def register_client(websocket):
    """Handle new connections."""
    CONNECTED_CLIENTS.add(websocket)
    print(f"Client connected. Total: {len(CONNECTED_CLIENTS)}")
    try:
        await websocket.wait_closed()
    finally:
        CONNECTED_CLIENTS.remove(websocket)
        print(f"Client disconnected. Total: {len(CONNECTED_CLIENTS)}")

def construct_payload(hands_contract, timestamp_ms, image_frame):
    """Bundles Spatial Data + Video Frame into one JSON packet."""
    
    # 1. Compress Image to JPEG
    retval, buffer = cv2.imencode('.jpg', image_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    b64_image = base64.b64encode(buffer).decode('utf-8')
    
    # 2. Build Payload
    payload = {
        "timestamp": timestamp_ms,
        # "image": "data:image/jpeg;base64," + b64_image,
        "image": None,  
        "hands": hands_contract
    }
    return json.dumps(payload)

async def main():
    force_camera_settings()
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    tracker = HandTracker()
    visualizer = DebugVisualizer()
    smoother = LandmarkSmoother(alpha=0.6)
    mapper = CoordinateMapper(WIDTH, HEIGHT, z_close=4.0, z_far=10.0)

    print(f"--- HEADLESS SENSOR NODE RUNNING ---")
    print(f"--- Listening on ws://localhost:8765 ---")

    async with websockets.serve(register_client, "localhost", 8765):
        prev_time = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # CRITICAL: Allow async loop to breathe
            await asyncio.sleep(0.001) 

            ts_ms = int(time.time() * 1000)
            tracker.detect_async(frame, ts_ms)
            result = tracker.get_latest_result()

            hands_contract = []
            hands_radar = []

            if result and result.hand_landmarks:
                for idx, raw_lms in enumerate(result.hand_landmarks):
                    label = result.handedness[idx][0].category_name if len(result.handedness) > idx else "Unknown"
                    smoothed = smoother.smooth(raw_lms, label)
                    
                    # Depth Logic
                    wrist, idx_mcp = smoothed[0], smoothed[5]
                    depth_proxy = 1.0 / (((wrist.x - idx_mcp.x)**2 + (wrist.y - idx_mcp.y)**2)**0.5 + 0.01)
                    
                    # Map Full Skeleton
                    mapped_skel = []
                    for lm in smoothed:
                        wx, wy, wz = mapper.normalize(lm.x, lm.y, depth_proxy)
                        mapped_skel.append({"x": round(wx, 4), "y": round(wy, 4), "z": round(wz, 4)})
                    
                    hands_contract.append({"id": label, "landmarks": mapped_skel})
                    
                    # Radar Data
                    w_pt = mapped_skel[0]
                    hands_radar.append({'label': label, 'x': w_pt['x'], 'y': w_pt['y'], 'z': w_pt['z']})

            # Draw Radar on frame (Conceptually, this is "Augmented Reality" generated on server)
            # Since we are headless, this 'debug_frame' is what the client sees.
            debug_frame = visualizer.draw_radar(frame, hands_radar)
            
            # Broadcast
            if CONNECTED_CLIENTS:
                payload = construct_payload(hands_contract, ts_ms, debug_frame)
                tasks = [asyncio.create_task(ws.send(payload)) for ws in CONNECTED_CLIENTS]

            # NO cv2.imshow() -> Completely Headless

    cap.release()

if __name__ == "__main__":
    asyncio.run(main())