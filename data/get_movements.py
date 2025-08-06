import os
import time
import random
import requests
import pandas as pd

stations_df = pd.read_csv("data/csv/station_ids.csv")

# API configuration
base_url = "https://wiediversistmeingarten.org/api"
amount_movements = 100000
output_csv_path = "data/csv/all_validated_movements.csv"
os.makedirs("data/csv", exist_ok=True)

# Manually crafted list of invalid/incorrect class labels
excluded_validations = [
    "cyanocitta cristata", "poecile atricapillus", "sitta canadensis", "melospiza lincolnii", "hirundo rustica",
    "molothrus ater", "thryomanes bewickii", "thryothorus ludovicianus", "poecile carolinensis",
    "ardea herodias", "anthornis melanura", "hemiphaga novaeseelandiae", "seiurus aurocapilla",
    "egretta novaehollandiae", "alopochen aegyptiaca", "sitta pygmaea", "corvus corax", "familie",
    "dohle", "no bird", "grünfink", "gruenfink", "blaumeise", "sumpf", "deutsches",
    "kohlmeise", "erlenzeisig", "weidenmeise", "sumpfmeise", "erlen", "test", "b",
    "stofftier", "unscharf", "eichhörnchen", "eichelhäher", "buchfink", "mönchsgrasmücke",
    "elster", "kernbeißer, star", "kohlemeise", "erlej", "gimpel", "human", "kohl",
    "haussperling", "blau", "cloris", "buntsprecht", "parus major u. feldsperling"
]

all_validated_data = []

# Helper: Fetch with retries and timeout
def fetch_with_retries(url, max_retries=3, base_delay=2):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response
            else:
                print(f"[WARN] Status {response.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Attempt {attempt + 1} failed: {e}")
        time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
    return None

# Iterate over stations
for _, row in stations_df.iterrows():
    station_id = row["station_id"]
    station_name = str(row["name"])

    print(f"[INFO] Fetching movements for station {station_name} ({station_id})")
    movements_url = f"{base_url}/movement/{station_id}?movements={amount_movements}"
    response = fetch_with_retries(movements_url)

    if response is None:
        print(f"[EXCEPTION] Station {station_id} failed after retries")
        continue

    try:
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
                name = (val.get("latinName", "") or val.get("germanName", "")).strip().lower()
                if not name or name == "none":
                    continue

                if any(excluded in name for excluded in excluded_validations):
                    continue
                names.add(name)

            if not names and len(validations) == 1:
                names.add("verify")

            if not names:
                continue

            detections = mov.get("detections", {})
            predictions = set()
            for det in detections:
                name = det.get("latinName", "").strip().lower()
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

    # Delay between stations
    time.sleep(random.uniform(1.0, 2.5))

# Save collected data
if all_validated_data:
    df = pd.DataFrame(all_validated_data)
    df.to_csv(output_csv_path, index=False)
    print(f"[OK] Saved all validated movements to {output_csv_path}")
else:
    print("[INFO] No validated movements collected from any station")


from collections import Counter

# Summary statistics
species_counter = Counter()

try:
    df = pd.read_csv(output_csv_path)
    if "validations" in df.columns:
        species_counter.update(df["validations"].dropna().tolist())
except Exception as e:
    print(f"[ERROR] Failed to process validation summary: {e}")

print("\nTotal validated movements per bird species:")
for species, count in species_counter.most_common():
    print(f"{species}: {count}")
