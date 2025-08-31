import argparse
from pathlib import Path
import pandas as pd
import requests


def download_video(session, video_url, output_path):
    """Download the movements video using the same session."""
    try:
        response = session.get(video_url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8129):
                    f.write(chunk)
            print(f"[OK] Downloaded video to {output_path}")
        else:
            print(f"[ERROR] Failed to download {output_path}: HTTP {response.status_code}")
    except Exception as e:
        print(f"[EXCEPTION] Error downloading {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download the videos from a list of movements.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory.")
    parser.add_argument("--user-agent", type=str, default="", help="User agent for requests.")
    parser.add_argument("--input", type=str, default="movements.csv", help="Name of the file containing the movements.")
    parser.add_argument("--use-sample", type=int, default=10, help="Path to local data directory.")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    output_dir = workdir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the CSV
    movs_csv = workdir / args.input
    if not movs_csv.exists():
        raise FileNotFoundError(f"Missing {movs_csv}!")
    mov_df = pd.read_csv(movs_csv)

    # Use only a sample
    if args.use_sample > 0:
        mov_df = mov_df.groupby("validations", group_keys=False).apply(lambda x: x.sample(min(len(x), args.use_sample), random_state=42))
        mov_df.to_csv(workdir / "sampled_movs.csv", index=False)
        print(f"Saved {len(mov_df.index)} sampled movements to {workdir / 'sampled_movs.csv'}.")

    with requests.Session() as session:
        if args.user_agent:
            # Create a session with custom User-Agent
            session.headers.update({"User-Agent": args.user_agent})

        # Iterate over rows and download each video
        for _, row in mov_df.iterrows():
            station_id = row["station_id"]
            mov_id = row["mov_id"]
            video_url = row["video_link"]
            species_name = row["validations"].strip().replace(" ", "_")

            # Create species-specific subdirectory
            species_dir = output_dir / species_name
            species_dir.mkdir(parents=True, exist_ok=True)

            # Create a filename based on station and movement ID
            filename = f"{station_id}_{mov_id}.mp4"
            output_path = species_dir / filename

            # Download video using the session
            download_video(session, video_url, output_path)


if __name__ == "__main__":
    main()