import requests
import pandas as pd

# Load stations
stations_df = pd.read_csv("data/csv/station_ids.csv")

base_url = "https://wiediversistmeingarten.org/api"

# To speed up set a limit of returned movements per station
amount_movements = 1000

all_validated_data = []

for _, row in stations_df.iterrows():
    station_id = row["station_id"]
    station_name = row["name"]

    try:
        # Fetch movements for the station
        movements_url = f"{base_url}/movement/{station_id}?movements={amount_movements}"
        response = requests.get(movements_url)
        if response.status_code != 200:
            print(f"[ERROR] Could not fetch movements for station {station_id}")
            continue

        movements = response.json()
        if not movements:
            # Skip if no movements
            print(f"[INFO] No movements for station: {station_name}")
            continue

        # Filter validated movements and collect mov_id and latinName
        validated_data = []
        for mov in movements:
            validations = mov.get("validation", {}).get("validations", [])
            if not validations:
                continue

            names = set()
            for val in validations:
                name = (val.get("latinName") or val.get("germanName")).strip().lower()
                if name and name != "none":
                    names.add(name)

            detections = mov.get("detections", {})
            predictions = set()
            for det in detections:
                name = det.get("latinName").lower()
                score = det.get("score")
                if name and score is not None:
                    predictions.add((name, round(score, 2)))

            if names:
                validated_data.append({
                    "station_id":   station_id,
                    "mov_id":       mov["mov_id"],
                    "predictions":  "; ".join(str(t) for t in sorted(predictions, key=lambda x: x[1], reverse=True)),
                    "validations":  "; ".join(str(t) for t in sorted(names)),
                    "video_link":   mov["video"],
                })

        if validated_data:
            all_validated_data.extend(validated_data)
            print(f"[OK] Collected {len(validated_data)} validated movement(s) for: {station_name}")
        else:
            print(f"[INFO] No validated movements for station: {station_name}")

    except Exception as e:
        print(f"[EXCEPTION] Error processing station {station_id}: {e}")

if all_validated_data:
    df = pd.DataFrame(all_validated_data)
    df.to_csv(f"data/csv/all_validated_movements.csv", index=False)
    print("[OK] Saved all validated movements to all_validated_movements.csv")
else:
    print("[INFO] No validated movements collected from any station")


from collections import Counter

species_counter = Counter()

try:
    df = pd.read_csv("data/all_validated_movements.csv")
    if "validations" in df.columns:
        species_counter.update(df["validations"].dropna().tolist())
except Exception as e:
    print(f"[ERROR] Failed to process: {e}")

# Print total count per species
print("\nTotal validated movements per bird species:")
for species, count in species_counter.most_common():
    print(f"{species}: {count}")
