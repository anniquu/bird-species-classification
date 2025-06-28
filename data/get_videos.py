import pandas as pd
import os
import requests

# Ensure output directory exists
output_dir = "data/videos"
os.makedirs("data/videos", exist_ok=True)

# Load the CSV
mov_df = pd.read_csv("data/csv/all_validated_movements.csv")

sampled_df = mov_df.groupby("validations", group_keys=False).apply(lambda x: x.sample(min(len(x), 10), random_state=42))

# Iterate over rows and download each video
for _, row in sampled_df.iterrows():
    station_id = row["station_id"]
    mov_id = row["mov_id"]
    video_url = row["video_link"]
    species_name = row["validations"].strip().replace(" ", "_")

    # Create species-specific subdirectory
    species_dir = os.path.join(output_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)

    # Create a filename based on station and movement ID
    filename = f"{station_id}_{mov_id}.mp4"
    filepath = os.path.join(species_dir, filename)

    try:
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[OK] Downloaded video: {filename} to {species_name}/")
        else:
            print(f"[ERROR] Failed to download {filename}: HTTP {response.status_code}")
    except Exception as e:
        print(f"[EXCEPTION] Error downloading {filename}: {e}")
