import time
import asyncio
from typing import List, Dict, Any, Set

# Local Module Imports
from Capture import Capture, WIDTH, HEIGHT
from CoordinateMapper import CoordinateMapper
from GestureEngine import GestureEngine
from HandTracker import HandTracker
from LandmarkSmoother import LandmarkSmoother
from Server import Server

# Network Configuration
HOST: str = 'localhost'
PORT: int = 8765

async def main() -> None:
    """
    The main execution loop for the Touchless Interaction Node.
    
    Architecture:
        1. Capture: Grabs frames from the webcam (Synchronous/Blocking CV2).
        2. Detection: Sends frames to MediaPipe (Asynchronous/Callback-based).
        3. Logic: Smooths data, calculates depth, and detects gestures.
        4. Broadcast: Sends JSON payloads to connected WebSocket clients (Asynchronous).
    """

    # --- 1. COMPONENT INITIALIZATION ---
    tracker = HandTracker()
    
    # Smoothing factor: 0.6 balances jitter reduction vs. latency
    smoother = LandmarkSmoother(alpha=0.6)
    
    # Mapper converts 2D camera coordinates -> 3D Interaction space
    mapper = CoordinateMapper(WIDTH, HEIGHT, z_close=2.0, z_far=6.0)
    
    # Gesture engine with hysteresis thresholds (0.05 pinch / 0.08 release)
    gestures = GestureEngine(pinch_threshold=0.05, release_threshold=0.08)
    
    server = Server()
    
    # Initialize Camera Hardware
    try:
        capture = Capture()
    except IOError as e:
        print(f"CRITICAL ERROR: {e}")
        return

    print(f"--- INTERACTION NODE RUNNING on {HOST}:{PORT} ---")

    # --- 2. MAIN EVENT LOOP ---
    try:
        # Start the WebSocket server context
        async with server.serve(server.register_client, HOST, PORT):
            
            # Keep track of background tasks (sending messages) to prevent garbage collection
            background_tasks: Set[asyncio.Task] = set()

            while True:
                # A. ACQUIRE FRAME
                # Note: This is a blocking call. If the camera is slow, the loop slows down.
                ret, frame = capture.read()
                if not ret or frame is None: 
                    print("Warning: Failed to grab frame.")
                    break
                
                # B. YIELD CONTROL
                # Crucial: asyncio.sleep(0) or small delay allows the Event Loop to 
                # process pending WebSocket pings/pongs/connections.
                await asyncio.sleep(0.001) 

                ts_ms = int(time.time() * 1000)
                
                # C. DETECTION (MEDIA PIPE)
                # We send the frame to the graph. Results are updated via callback in HandTracker.
                tracker.detect_async(frame, ts_ms)
                result = tracker.get_latest_result()

                hands_contract: List[Dict[str, Any]] = []

                # D. PROCESSING PIPELINE
                if result and result.hand_landmarks:
                    for idx, raw_lms in enumerate(result.hand_landmarks):
                        
                        # Safety: Ensure handedness labels exist for this index
                        if len(result.handedness) > idx:
                            # 'category_name' is usually "Left" or "Right"
                            label = result.handedness[idx][0].category_name
                        else:
                            label = "Unknown"

                        # D1. Smoothing (Low Pass Filter)
                        smoothed = smoother.smooth(raw_lms, label)
                        
                        # D2. Depth Proxy Calculation
                        # We estimate Z-depth by measuring the distance between Wrist (0) and Index MCP (5).
                        # As the hand gets closer to the camera, this distance in 2D pixels increases.
                        wrist, idx_mcp = smoothed[0], smoothed[5]
                        dist_sq = (wrist.x - idx_mcp.x)**2 + (wrist.y - idx_mcp.y)**2
                        # Inverse relationship: Larger size = Smaller Z (Closer)
                        depth_proxy = 1.0 / ((dist_sq**0.5) + 0.01)
                        
                        # D3. Gesture Recognition
                        is_pinching = gestures.detect_pinch(smoothed, label)

                        # D4. Coordinate Mapping (Normalization)
                        mapped_skel = []
                        for lm in smoothed:
                            # Convert 0..1 image space to -1..1 World Space
                            wx, wy, wz = mapper.normalize(lm.x, lm.y, depth_proxy)
                            mapped_skel.append({
                                "x": round(wx, 4), 
                                "y": round(wy, 4), 
                                "z": round(wz, 4)
                            })
                        
                        # Construct the data packet for this hand
                        hands_contract.append({
                            "id": label, 
                            "pinch": is_pinching,  # Boolean trigger for "Click/Drag"
                            "landmarks": mapped_skel
                        })

                # E. BROADCASTING
                # Only send data if we have clients and we detected hands
                if server.CONNECTED_CLIENTS and hands_contract:
                    payload = server.construct_payload(hands_contract, ts_ms)
                    
                    # Fire-and-Forget pattern:
                    # We create tasks to send data so we don't block the next camera frame 
                    # while waiting for network I/O.
                    for ws in server.CONNECTED_CLIENTS:
                        task = asyncio.create_task(ws.send(payload))
                        background_tasks.add(task)
                        # Remove task from set when done to free memory
                        task.add_done_callback(background_tasks.discard)

    except KeyboardInterrupt:
        print("\nStopping Interaction Node...")
    finally:
        # F. CLEANUP
        # Ensure the camera resource is freed back to the OS
        capture.release()
        print("Camera released.")

if __name__ == "__main__":
    asyncio.run(main())