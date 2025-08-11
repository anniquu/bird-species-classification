import argparse
from pathlib import Path
import pandas as pd
import requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the videos from a list of movements.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory.")
    parser.add_argument("--input", type=str, default="movements.csv", help="Name of the file containing the movements.")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    output_dir = workdir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the CSV
    mov_df = pd.read_csv(workdir / args.input)

    # Use only a sample (for testing)
    # sampled_df = mov_df.groupby("validations", group_keys=False).apply(lambda x: x.sample(min(len(x), 10), random_state=42))

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

        try:
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                with open(species_dir / filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"[OK] Downloaded video: {filename} to {species_name}/")
            else:
                print(f"[ERROR] Failed to download {filename}: HTTP {response.status_code}")
        except Exception as e:
            print(f"[EXCEPTION] Error downloading {filename}: {e}")
