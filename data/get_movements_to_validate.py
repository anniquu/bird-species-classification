import requests
import pandas as pd

# Load stations
stations_df = pd.read_csv("data/csv/station_ids.csv")

base_url = "https://wiediversistmeingarten.org/api"

# To speed up set a limit of returned movements per station
amount_movements = 100000

all_movements = []

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
            mov_id = mov["mov_id"]

            validations = mov.get("validation", {}).get("validations", [])
            if validations:
                continue

            detections = mov.get("detections", {})
            predictions = set()
            for det in detections:
                name = det.get("latinName").lower()
                score = det.get("score")
                if name and score is not None:
                    predictions.add((name, round(score, 2)))

            validated_data.append({
                "predictions":      "; ".join(str(t) for t in sorted(predictions, key=lambda x: x[1], reverse=True)) if predictions else "No predictions",
                "validation_link":  f"https://wiediversistmeingarten.org/view/station/{station_id}/{mov_id}",
            })

        if validated_data:
            all_movements.extend(validated_data)
            print(f"[OK] Collected {len(validated_data)} movement(s) for: {station_name}")
        else:
            print(f"[INFO] No movements for station: {station_name}")

    except Exception as e:
        print(f"[EXCEPTION] Error processing station {station_id}: {e}")

if all_movements:
    df = pd.DataFrame(all_movements).sort_values("predictions")
    df.to_csv(f"data/csv/movements_to_validate.csv", index=False)
    print("[OK] Saved all movements to movements_to_validate.csv")
else:
    print("[INFO] No movements collected from any station")