# SpatialOS (v1.0)

### 1. What This Is
A hardware-agnostic spatial input framework that transforms standard 2D webcams into 6DoF interaction devices. It provides a headless backend for skeletal tracking, a real-time WebSocket transport layer, and a browser-based runtime for bi-manual 3D manipulation. It solves the problem of enabling high-fidelity spatial computing without specialized VR/AR hardware.

### 2. Mental Model
The system operates as a unidirectional pipeline with strict separation of concerns:
* **Sensor Node (Python):** Acts as the hardware driver. It captures raw video, performs ML inference (MediaPipe), stabilizes coordinates, and broadcasts a "truth" state. It does not render visuals.
* **The Spatial Contract:** A JSON-over-WebSocket stream carrying normalized, world-space skeletal data (30Hz). This is the only link between perception and reality.
* **Spatial Runtime (Three.js):** A rendering engine that consumes the contract. It maps raw data into a `SceneGraph` but owns the *interpretation* of that data.
* **Interaction Stack:** Input flows through a hierarchical filter: `Mode Manager` → `Active Tool` → `Scene Entities`.



### 3. Invariants (The Laws)
Violating these breaks the system:
1.  **Coordinate System:** Right-handed Y-up. Unit scale is approx 10cm. Origin (0,0,0) is the center of the tracking frustum.
2.  **Backend Authority:** The Python node owns the *existence* of hands. The Browser owns the *intent* of hands.
3.  **Data Contract:** Frame packets must always contain `timestamp`, `id`, `pinch`, and `landmarks[21]`.
4.  **Resolution Order:** Global Mode > Active Tool > Hand State > Object Collision.
5.  **Persistence:** State is serialized to `localStorage` on a dirty-write cycle (5s). On load, entity hydration order is `World` → `Tools` → `State`.

### 4. What This Is Not
* **Not a Gesture Library:** It does not detect "peace signs" or "thumbs up." It tracks position and pinch state only.
* **Not a WebXR Polyfill:** It does not use the WebXR Device API and is not compatible with headset browsers.
* **Not a UI Library:** It does not provide 2D HTML overlays. All UI is strictly 3D geometry (spatial panels).

### 5. Minimal Setup
**Backend:**
```bash
pip install opencv-python mediapipe websockets
python sensor.py
````

**Frontend:**

1.  Serve the directory (e.g., VS Code "Live Server" or `python -m http.server`).
2.  Open `client.html` in a desktop browser (Chrome/Firefox).
3.  Verify the HUD badge reads "ONLINE".

### 6\. Folder Structure

```text
/
├── sensor.py             # The Headless Perception Engine (Run this first)
├── client.html           # The Spatial Runtime & Application Logic
├── hand_landmarker.task  # MediaPipe Model Binary
└── README.md             # This file
```

### 7\. Extension Points

  * **New Tools:** Extend the `Tool` class in `client.html` and register via `os.registerTool()`.
  * **New Interactive Objects:** Create a class with a `check(pos)` method and `serialize/hydrate` methods. Register via `os.persistence.register()`.
  * **New Sensors:** Modify `sensor.py` to ingest different hardware (e.g., Depth Cam) as long as it outputs the **Spatial Contract**.
  * **DO NOT TOUCH:** The `SpatialOS` class `processFrame()` loop or the WebSocket handshake logic.

### 8\. Non-Goals

  * **Multi-User Sync:** The current architecture assumes a single local observer.
  * **Mobile Support:** The runtime is optimized for desktop rendering pipelines.
  * **Security:** There is no auth layer between the Sensor Node and the Runtime. Localhost only.
  * **Haptics:** No support for physical feedback devices.

<!-- end list -->

```
```