import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, filedialog, Button
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def select_video_file():
    """Open a file dialog to select a video file."""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.flv")]
    )
    root.destroy()
    return file_path

def process_video(video_path):
    """Process the video for pose estimation."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file at path: {video_path}")
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                logging.info("End of video file or cannot read the frame.")
                break

            # Convert frame to RGB for MediaPipe Pose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # If pose is detected, draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame
            cv2.imshow('Pose Detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                break
    
    cap.release()
    cv2.destroyAllWindows()

def process_video_with_multiple_angles(video_path, angles=[0, 90, 180, 270]):
    """Process video for pose estimation with retries at multiple angles."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file at path: {video_path}")
    
    for angle in angles:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the first frame
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    logging.info("End of video file or cannot read the frame.")
                    break
                
                # Rotate the frame by the given angle
                frame = rotate_image(frame, angle)
                
                # Convert frame to RGB for MediaPipe Pose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                # If pose is detected, draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display the frame
                cv2.imshow(f'Pose Detection at {angle} degrees', frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                    break
    
    cap.release()
    cv2.destroyAllWindows()

def rotate_image(image, angle):
    """Rotate the image by a specific angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def process_video_file():
    """Process video file for pose estimation and contour detection."""
    video_path = select_video_file()

    if not video_path:
        logging.info("No video selected.")
        return  # Exit if no video is selected

    try:
        # Process video for pose estimation
        process_video(video_path)
    except FileNotFoundError as e:
        logging.error(e)

def main():
    """Main function to run enhanced detection and visualization for video files."""
    root = Tk()
    root.title("Enhanced Detection")
    root.geometry("200x100")  # Set the window size

    def on_retry():
        """Handle the retry button click event."""
        try:
            process_video_file()
        except FileNotFoundError as e:
            logging.info(e)

    # Initial processing of the video file
    try:
        process_video_file()
    except FileNotFoundError as e:
        logging.info(e)

    # Create a Retry button
    retry_button = Button(root, text="Retry", command=on_retry)
    retry_button.pack(pady=20)  # Adjust button placement

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
