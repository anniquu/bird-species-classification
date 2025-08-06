import os
import cv2
import glob

def center_crop_to_square(frame):
    """Crop the center square from a frame."""
    height, width, _ = frame.shape
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Input and output base directories
video_base_dir = "data/videos"
frame_base_dir = "data/frames"

# Create output base directory if not exists
os.makedirs(frame_base_dir, exist_ok=True)

# Traverse species directories
for species in os.listdir(video_base_dir):
    species_dir = os.path.join(video_base_dir, species)
    if not os.path.isdir(species_dir):
        continue

    for video_file in glob.glob(os.path.join(species_dir, "*.mp4")):
        video_filename = os.path.basename(video_file)
        video_id = os.path.splitext(video_filename)[0]

        output_dir = os.path.join(frame_base_dir, species)
        os.makedirs(output_dir, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"[ERROR] Could not open {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) * 1  # Capture all frames

        frame_count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                cropped = center_crop_to_square(frame)
                frame_path = os.path.join(output_dir, f"{video_id}_{saved_count:03d}.jpg")
                cv2.imwrite(frame_path, cropped)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f"[OK] Extracted {saved_count} frames from {video_filename} into {output_dir}")
