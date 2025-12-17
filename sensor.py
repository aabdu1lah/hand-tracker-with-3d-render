import cv2
import mediapipe as mp
import time
import os
import numpy as np
import asyncio
import websockets
import json
import base64
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = 'hand_landmarker.task'
WEBCAM_ID = 0
WIDTH, HEIGHT = 1280, 720
TARGET_FPS = 30
JPEG_QUALITY = 40 

# --- CLASSES ---

class GestureEngine:
    """Analyzes hand geometry to detect intent (Pinches, etc)."""
    def __init__(self, pinch_threshold=0.05, release_threshold=0.07):
        self.pinch_thresh = pinch_threshold
        self.release_thresh = release_threshold
        self.state = {'Left': False, 'Right': False} # True = Pinching

    def detect(self, landmarks, label):
        # 1. Get Thumb Tip (4) and Index Tip (8)
        thumb = landmarks[4]
        index = landmarks[8]
        
        # 2. Calculate Euclidean Distance
        # Note: These are normalized coordinates (0.0-1.0), so 0.05 is roughly 5% of screen width
        dist = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2 + (thumb.z - index.z)**2)**0.5
        
        # 3. Hysteresis Logic (Prevents flickering)
        if self.state.get(label, False):
            # Currently pinching, check for release
            if dist > self.release_thresh:
                self.state[label] = False
        else:
            # Currently open, check for pinch
            if dist < self.pinch_thresh:
                self.state[label] = True
                
        return self.state[label]

class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_landmarks = {} 

    def smooth(self, current_landmarks, hand_label):
        if hand_label not in self.prev_landmarks:
            self.prev_landmarks[hand_label] = current_landmarks
            return current_landmarks
        smoothed_hand = []
        prev_hand = self.prev_landmarks[hand_label]
        for i, curr_lm in enumerate(current_landmarks):
            prev_lm = prev_hand[i]
            new_x = (curr_lm.x * self.alpha) + (prev_lm.x * (1 - self.alpha))
            new_y = (curr_lm.y * self.alpha) + (prev_lm.y * (1 - self.alpha))
            new_z = (curr_lm.z * self.alpha) + (prev_lm.z * (1 - self.alpha))
            class SmoothedLM:
                def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
            smoothed_hand.append(SmoothedLM(new_x, new_y, new_z))
        self.prev_landmarks[hand_label] = smoothed_hand
        return smoothed_hand

class CoordinateMapper:
    def __init__(self, width, height, z_close=4.0, z_far=10.0):
        self.width, self.height = width, height
        self.aspect_ratio = width / height
        self.z_min, self.z_max = z_close, z_far

    def normalize(self, x_raw, y_raw, z_raw):
        # --- THE FLIP IS HERE ---
        # Old: x = (x_raw - 0.5) * self.aspect_ratio
        # New: We negate the direction so Camera Left becomes World Right
        x = - (x_raw - 0.5) * self.aspect_ratio
        
        # Depth mapping
        z_clamped = max(self.z_min, min(z_raw, self.z_max))
        z_norm = (z_clamped - self.z_min) / (self.z_max - self.z_min)
        z = z_norm * 2 - 1
        
        # Y-flip remains (Image Down = World Down)
        # Note: In 3D, Y is Up. Image Y increases Down. 
        # So -(y-0.5) correctly maps Top-Image to Top-World.
        y = - (y_raw - 0.5) 
        
        return x, y, z

class HandTracker:
    def __init__(self, model_path=MODEL_PATH):
        self.result = None
        self.latest_timestamp_ms = 0
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.3, min_tracking_confidence=0.3,
            result_callback=self.print_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def print_result(self, result, output_image, timestamp_ms):
        self.result = result

    def detect_async(self, frame, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        if timestamp_ms > self.latest_timestamp_ms:
            self.landmarker.detect_async(mp_image, timestamp_ms)
            self.latest_timestamp_ms = timestamp_ms
    
    def get_latest_result(self): return self.result

# --- NETWORK LOGIC ---

CONNECTED_CLIENTS = set()

async def register_client(websocket):
    CONNECTED_CLIENTS.add(websocket)
    try: await websocket.wait_closed()
    finally: CONNECTED_CLIENTS.remove(websocket)

def construct_payload(hands_contract, timestamp_ms):
    # We removed the image payload for Step 4 to focus on interaction speed
    # (Optional: Add image back if you really need the hud)
    payload = {
        "timestamp": timestamp_ms,
        "hands": hands_contract
    }
    return json.dumps(payload)

def force_camera_settings():
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0")
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=3")

# --- MAIN ---

async def main():
    force_camera_settings()
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    tracker = HandTracker()
    smoother = LandmarkSmoother(alpha=0.6)
    mapper = CoordinateMapper(WIDTH, HEIGHT, z_close=2.0, z_far=6.0)
    gestures = GestureEngine(pinch_threshold=0.05, release_threshold=0.08)

    print(f"--- INTERACTION NODE RUNNING ---")

    async with websockets.serve(register_client, "localhost", 8765):
        while True:
            ret, frame = cap.read()
            if not ret: break
            await asyncio.sleep(0.001) 

            ts_ms = int(time.time() * 1000)
            tracker.detect_async(frame, ts_ms)
            result = tracker.get_latest_result()

            hands_contract = []

            if result and result.hand_landmarks:
                for idx, raw_lms in enumerate(result.hand_landmarks):
                    label = result.handedness[idx][0].category_name if len(result.handedness) > idx else "Unknown"
                    smoothed = smoother.smooth(raw_lms, label)
                    
                    # 1. Depth Calculation
                    wrist, idx_mcp = smoothed[0], smoothed[5]
                    depth_proxy = 1.0 / (((wrist.x - idx_mcp.x)**2 + (wrist.y - idx_mcp.y)**2)**0.5 + 0.01)
                    
                    # 2. Gesture Detection (The new brain)
                    is_pinching = gestures.detect(smoothed, label)

                    # 3. Map Skeleton
                    mapped_skel = []
                    for lm in smoothed:
                        wx, wy, wz = mapper.normalize(lm.x, lm.y, depth_proxy)
                        mapped_skel.append({"x": round(wx, 4), "y": round(wy, 4), "z": round(wz, 4)})
                    
                    hands_contract.append({
                        "id": label, 
                        "pinch": is_pinching,  # <--- AUTH STATE
                        "landmarks": mapped_skel
                    })

            if CONNECTED_CLIENTS and hands_contract:
                payload = construct_payload(hands_contract, ts_ms)
                tasks = [asyncio.create_task(ws.send(payload)) for ws in CONNECTED_CLIENTS]

    cap.release()

if __name__ == "__main__":
    asyncio.run(main())