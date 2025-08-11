import argparse
from pathlib import Path
import cv2
import glob


def center_crop_to_square(frame):
    """Crop the center square from a frame."""
    height, width, _ = frame.shape
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def main():
    parser = argparse.ArgumentParser(description="Split videos into frames.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory.")
    parser.add_argument("--frame-rate", type=float, default=None,
        help="Number of frames-per-second to capture (default: capture all frames).")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Input and output base directories
    video_base_dir = workdir / "videos"
    frame_base_dir = workdir / "frames"
    frame_base_dir.mkdir(parents=True, exist_ok=True)

    # Traverse species directories
    for species_dir in video_base_dir.iterdir():
        if not species_dir:
            continue

        output_dir = frame_base_dir / species_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over mp4 files in the species directory
        for video_path in species_dir.glob("*.mp4"):
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[ERROR] Could not open {video_path}")
                continue


            # Determine frame capture interval
            if args.frame_rate and args.frame_rate > 0:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30   # Get frame rate or default to 30 fps
                frame_interval = max(int(round(fps / args.frame_rate)), 1) # e.g. video frame rate 30, get 5 frames per second => save every 6th frame
            else:
                frame_interval = 1  # capture all frames

            frame_count = 0
            saved_count = 0
            while True:
                # Read next frame from video
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    cropped = center_crop_to_square(frame)
                    frame_path = output_dir / f"{video_path.stem}_{saved_count:03d}.jpg"
                    cv2.imwrite(str(frame_path), cropped)
                    saved_count += 1
                frame_count += 1

            cap.release()
            print(f"[OK] Extracted {saved_count} frames from {video_path.stem} into {output_dir}")


if __name__ == "__main__":
    main()
