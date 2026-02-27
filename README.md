# Computer Vision – 3D Pose Estimation Using OpenCV and MediaPipe

## Overview

This project applies 3D pose estimation to scenes from Thai horror films using OpenCV and MediaPipe. The objective was to evaluate body landmark detection performance in non-standard cinematic conditions such as low lighting, dramatic framing, and unconventional camera angles.

The system processes uploaded video clips frame-by-frame, extracts pose landmarks, and improves detection robustness through multi-angle frame rotation.

This project was developed as part of my Computer Vision course at Assumption University.

---

## Objectives

- Apply MediaPipe pose estimation to real-world film footage
- Analyze pose detection performance in complex visual environments
- Improve landmark detection accuracy using multi-angle frame processing
- Build a simple GUI-based workflow for user video input

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Tkinter

---

## System Architecture

1. User selects a video file via a Tkinter file dialog.
2. The video is processed frame-by-frame using OpenCV.
3. Each frame is passed to MediaPipe’s pose estimation model.
4. Body landmarks are extracted and visualized.
5. Frames are rotated at multiple angles (0°, 90°, 180°, 270°) and reprocessed to improve detection reliability.
6. Processed frames are displayed in real time.

---

## Core Components

### `select_video_file()`
Launches a Tkinter-based file dialog to allow the user to select a video.

### `process_video()`
Reads video frames, applies pose estimation, and renders landmark overlays.

### `rotate_image()`
Rotates frames to support multi-angle pose detection.

### `process_video_with_multiple_angles()`
Processes each frame at multiple orientations to improve landmark detection in challenging scenes.

---

## Dataset / Media

Nine clips were selected from Thai horror films featuring culturally significant ghost figures. The footage includes varied lighting, motion intensity, and camera orientation to test pose detection robustness under realistic cinematic conditions.

---

## How to Run

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

Run the program:

```bash
python ghostpositcode.py
```

A file selection window will appear. Choose a video clip to begin processing.

---

## Demonstration

The following video demonstrates:

- Uploading Thai horror film clips
- Real-time 3D pose landmark detection
- Multi-angle frame processing (0°, 90°, 180°, 270°)
- Visualization of MediaPipe body keypoints

[Watch the project demonstration](./Posit%20Demonstration%20(9%20Videos).mp4)

---
