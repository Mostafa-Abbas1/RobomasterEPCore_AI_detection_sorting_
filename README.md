# RoboMaster EP Core - AI Object Detection and Sorting

A Python project for the DJI RoboMaster EP robot that uses YOLOv8s for real-time object detection and autonomous sorting.

## Features

- **YOLOv8s Object Detection**: Real-time detection using the YOLOv8s model (small, fast, accurate)
- **Autonomous Sorting**: Automatically sorts detected objects into designated zones
- **MovementStack**: Records all movements for precise return to center position
- **Time-based Navigation**: Stable zone navigation using drive_speed() for drift-free movement
- **Live Camera Feed**: Stream and process video from the robot camera
- **Gripper Control**: Pick and place objects using the robotic arm and gripper

## Project Structure

    dji-robo-master-ep-core/
    |-- src/
    |   |-- robot_control/          # Robot hardware control
    |   |   |-- connection.py       # Robot connection management
    |   |   |-- movement.py         # Movement, MovementStack
    |   |   |-- camera.py           # Camera streaming
    |   |   |-- gripper.py          # Gripper control
    |   |   |-- scanner.py          # Object scanning
    |   |   +-- threaded_camera.py  # Threaded camera for performance
    |   |
    |   |-- vision/                 # Computer vision
    |   |   |-- detection.py        # YOLOv8s object detection
    |   |   |-- preprocessing.py    # Image preprocessing
    |   |   +-- tracking.py         # Object tracking
    |   |
    |   +-- sorting/                # Sorting logic
    |       |-- logic.py            # Sorting controller (time-based navigation)
    |       |-- strategy.py         # Sorting strategies
    |       +-- zones.py            # Zone management
    |
    |-- config/
    |   +-- settings.py             # Configuration settings
    |
    |-- tests/
    |   |-- live_yolov8s_camera.py  # Live camera detection test
    |   +-- examples/               # RoboMaster SDK examples
    |
    |-- main.py                     # Main sorting program
    |-- demo_sorting_floor.py       # Floor sorting demo
    |-- yolov8s.pt                  # YOLOv8s model (downloaded on first run)
    |-- requirements.txt            # Python dependencies
    +-- README.md

## Requirements

- Python 3.8+
- DJI RoboMaster EP robot
- WiFi connection to robot

## Installation

1. Clone the repository
2. Create a virtual environment (recommended)
3. Install dependencies: pip install -r requirements.txt

The YOLOv8s model (yolov8s.pt) will be downloaded automatically on first run.

## Configuration

Edit config/settings.py to customize:

### Object Classes

Currently configured for 2 COCO dataset classes:
- bottle -> Zone B (RIGHT)
- cup -> Zone A (LEFT)

You can use any COCO dataset classes: cell phone, mouse, keyboard, banana, apple, etc.

### Sorting Zones

- zone_a: Turn LEFT (90 degrees) - for Cups
- zone_b: Turn RIGHT (-90 degrees) - for Bottles

### Main Parameters (in main.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| BRIGHTNESS_FACTOR | 1.5 | Image brightness for low light |
| BBOX_AREA_THRESHOLD | 50000 | When close enough to grab (higher = closer) |
| STEP_SIZE | 0.15 | Per-iteration approach distance (meters) |
| FINAL_APPROACH_DISTANCE | 0.35 | Final distance before grab (meters) |
| MAX_ITERATIONS | 10 | Maximum approach iterations |
| ROTATION_STEP | 15 | Degrees to rotate when searching (smaller = more careful) |

## Usage

### 1. Connect to Robot

1. Turn on the RoboMaster EP
2. Connect your computer to the robot WiFi network (via DJI app in Station mode)

### 2. Run Live Camera Test

Test YOLOv8s detection with live camera feed:

    python tests/live_yolov8s_camera.py

Controls:
- q or ESC: Quit
- s: Save screenshot
- +/-: Adjust confidence threshold

### 3. Run Main Sorting Program

    python main.py

The program will:
1. Connect to the robot
2. Start camera stream
3. Load YOLOv8s model
4. Scan for objects (bottle, cup)
5. Pick up detected objects
6. Sort into designated zones (LEFT for cups, RIGHT for bottles)

## How It Works

### Detection Pipeline

1. Camera Stream: Captures live video from robot camera (720p)
2. Image Enhancement: Brightness adjustment for low-light conditions
3. YOLOv8s Detection: Detects objects in each frame
4. Filtering: Only processes target classes (bottle, cup)

### Iterative Approach Algorithm

When an object is detected:
1. Calculate angle offset from frame center
2. Rotate to center object in view
3. Move forward by STEP_SIZE (0.15m)
4. Repeat until BBOX_AREA_THRESHOLD reached (object close enough)
5. Final approach of FINAL_APPROACH_DISTANCE (0.35m)

### MovementStack System

All movements during approach are recorded:
- Each rotation is saved: record(0, 0, angle)
- Each forward movement is saved: record(distance, 0, 0)

After grabbing, movements are reversed in LIFO order:
- Robot returns to exact center position
- No drift accumulation over multiple sorting cycles

### Time-based Zone Navigation

Zone navigation uses drive_speed() instead of move():
- More stable, drift-free movement
- Explicit stop commands between movements
- Calculated drive times based on speed and distance

Sorting Process:
1. Pick up object
2. Reverse all approach movements (MovementStack)
3. Rotate to target zone (LEFT or RIGHT)
4. Drive straight into zone
5. Release object
6. Reverse out of zone
7. Rotate back to center orientation

## Project Status

- [x] YOLOv8s object detection integrated
- [x] RoboMaster SDK fully integrated
- [x] Camera streaming and frame capture
- [x] Autonomous sorting implemented
- [x] Gripper control for pick and place
- [x] Zone-based navigation
- [x] Live camera test tool
- [x] MovementStack for precise positioning
- [x] Time-based navigation for stability

## RoboMaster SDK Documentation

https://robomaster-dev.readthedocs.io/en/latest/introduction.html

## Troubleshooting

### Connection Issues
- Ensure robot is powered on
- Check WiFi connection to robot
- Verify connection mode (Station mode recommended)

### Detection Issues
- Adjust DETECTION_CONFIDENCE_THRESHOLD in settings.py (lower = more detections)
- Adjust BBOX_AREA_THRESHOLD in main.py (higher = robot gets closer)
- Ensure good lighting conditions
- Check that objects are in the supported class list

### Movement Issues
- Robot not reaching object: Increase FINAL_APPROACH_DISTANCE
- Robot stopping too far: Increase BBOX_AREA_THRESHOLD
- Diagonal movement: Check drive_speed() is used (not move())

### Gripper Issues
- Verify robotic arm is attached
- Check gripper calibration
- Ensure objects are grippable size

## Acknowledgments

- DJI RoboMaster SDK
- Ultralytics YOLOv8
- OpenCV
