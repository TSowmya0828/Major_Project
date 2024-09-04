from argparse import ArgumentParser
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from utils import Box, get_position_ground, make_euler
from object_detector import ObjectDetector

# Transformation matrices and camera parameters
to_world_from_camera = np.identity(4)
pitch_matrix = make_euler(-27.39 * np.pi / 180, 0)
to_world_from_camera[:3, :3] = pitch_matrix
to_world_from_camera[:3, 3] = np.array([0, 8.5, 0])

# (from CARLA)
K = np.array([[770.19284237, 0.0, 640.0], [0.0, 770.19284237, 360.0], [0.0, 0.0, 1.0]])

def showbox(img, boxes):
    """
    Convert bounding boxes from pixel coordinates and dimensions that is normalized by depth
    to world bounding box
    """
    for box in boxes:
        # First find position in world
        position_on_ground = get_position_ground(
            box["x"], box["y"], K, to_world_from_camera, img.shape[0]
        )

        # Compute depth at this point (distance from camera)
        depth = np.linalg.norm(position_on_ground - to_world_from_camera[:3, 3])

        # Discard unrealistic bbox
        if box["w"] * depth < 1:
            continue

        if box["h"] * depth < 1:
            continue

        if box["l"] * depth < 1:
            continue

        # Multiply width/height/length by depth
        box = Box(
            position_on_ground[0],
            position_on_ground[2],
            box["w"] * depth,
            box["h"] * depth,
            box["l"] * depth,
        )
        # print(label)
        # Re-project them
        img = box.project(img, to_world_from_camera, K)

    return img

def process_video(video_path,frame_skip=1):
    object_detector = ObjectDetector(args.model, args.conf)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Error opening video stream or file")
        return
    frame_count=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count+=1
        if frame_count % frame_skip != 0:
            continue

        bboxs = object_detector.detect(frame)
        frame = showbox(frame, bboxs)

        cv2.imshow("Detection", frame)
        k = cv2.waitKey(1)

        if k == 27:
            break

def live_webcam_feed():
    object_detector = ObjectDetector(args.model, args.conf)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Error opening webcam feed")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Perform object detection
        bboxs = object_detector.detect(frame)
        # Show detected objects in 3D space
        frame = showbox(frame, bboxs)
        cv2.imshow("Webcam Feed with 3D Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_video(file_path)
    else:
        messagebox.showerror("Error", "No file selected.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object detection")
    parser.add_argument("model", type=str, help="Pytorch model for oriented cars bbox detection")
    parser.add_argument("--conf", type=float, default=0.5, help="Threshold to keep an object")
    args = parser.parse_args()

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Upload Image/Video")

    # Set the size of the window
    root.geometry("400x400")  # Width x Height

    # Change the background color of the window
    root.configure(bg="lightblue")

    # Create a label with custom text color and some padding
    label = tk.Label(root, text="Select an Image/Video to Process", bg="lightblue", fg="black")
    label.pack(side='top', pady=(100, 10))  # Add vertical padding and set side to top

    # Create a button to select a file
    select_button = tk.Button(root, text="Select Image/Video", command=select_file)
    select_button.pack(side='top', pady=10)  # Add vertical padding
 
    def start_detection():
        live_webcam_feed()

    # Create a button to start object detection using webcam feed
    start_button = tk.Button(root, text="Start Webcam Feed (Press 's')", command=start_detection)
    start_button.pack(side='top', pady=10)  # Add vertical padding

    # Start the Tkinter event loop
    root.mainloop()
