import argparse
from pathlib import Path
import time
import random
import requests
import pandas as pd


BASE_URL = "https://wiediversistmeingarten.org/api"
EXCLUDED_VALIDATIONS = [ # Manually crafted list of existing invalid/incorrect class labels
        "cyanocitta cristata", "poecile atricapillus", "sitta canadensis", "melospiza lincolnii", 
        "molothrus ater", "thryomanes bewickii", "thryothorus ludovicianus", "poecile carolinensis",
        "ardea herodias", "anthornis melanura", "hemiphaga novaeseelandiae", "seiurus aurocapilla",
        "egretta novaehollandiae", "alopochen aegyptiaca", "sitta pygmaea", "corvus corax", "familie",
        "dohle", "no bird", "grünfink", "gruenfink", "blaumeise", "sumpf", "deutsches", "kohl",
        "kohlmeise", "erlenzeisig", "weidenmeise", "sumpfmeise", "erlen", "test", "b","buchfink", 
        "stofftier", "unscharf", "eichhörnchen", "eichelhäher", "mönchsgrasmücke", "cloris",
        "elster", "kernbeißer, star", "kohlemeise", "erlej", "gimpel", "human", "haussperling",
        "blau", "buntsprecht", "parus major u. feldsperling","hirundo rustica"
    ]

def process_movements(station_id, mov, validated_only):
    """Extracts the validation(s) and predictions from a movement"""
    validations = mov.get("validation", {}).get("validations", [])
    if validated_only and not validations:
        return

    names = set()
    for val in validations:
        name = (val.get("latinName", "") or val.get("germanName", "")).strip().lower()
        if not name or name == "none":
            continue

        if any(excluded in name for excluded in EXCLUDED_VALIDATIONS):
            continue
        names.add(name)

    detections = mov.get("detections", {})
    predictions = set()
    for det in detections:
        name = det.get("latinName", "").strip().lower()
        score = det.get("score")
        if name and score is not None:
            predictions.add((name, round(score, 2)))

    return {
        "station_id": station_id,
        "mov_id": mov["mov_id"],
        "predictions": "; ".join(str(t) for t in sorted(predictions, key=lambda x: x[1], reverse=True)),
        "validations": "; ".join(str(t) for t in sorted(names)),
        "video_link": mov["video"],
    }


def fetch_with_retries(url, max_retries=5, base_delay=2):
    """Helper: Fetch with retries and timeout"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response
            else:
                print(f"[WARN] Status {response.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Attempt {attempt + 1} failed: {e}")
        time.sleep(base_delay * (2**attempt) + random.uniform(0, 1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch movements for stations and save to CSV.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory.")
    parser.add_argument("--validated-only", action=argparse.BooleanOptionalAction, 
        default=True, help="Only include movements with at least one human validation.")
    parser.add_argument("--number-movements", type=int, help="Number of movements to retrieve per station (API limit).")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    stations_csv = workdir / "station_ids.csv"
    if not stations_csv.exists():
        raise FileNotFoundError(f"Missing {stations_csv}!")
    stations_df = pd.read_csv(stations_csv)

    all_stations_movs = []

    # Iterate over stations
    for _, row in stations_df.iterrows():
        station_id = row["station_id"]
        station_name = str(row["name"])

        print(f"[INFO] Fetching movements for station {station_name} ({station_id})")

        # Build URL with optional limit
        movements_url = f"{BASE_URL}/movement/{station_id}"
        if args.number_movements:
            movements_url += f"?movements={args.number_movements}"

        response = fetch_with_retries(movements_url)
        if response is None:
            print(f"[EXCEPTION] Station {station_id} failed after retries")
            continue

        try:
            movements = response.json()
            if not movements:
                # Skip if the station has no movements
                print(f"[INFO] No movements for station: {station_name}")
                continue

            movement_data = []
            # Go through all of the movements of the station
            for mov in movements:
                processed = process_movements(station_id, mov, args.validated_only)
                if processed:
                    movement_data.append(processed)

            if movement_data:
                all_stations_movs.extend(movement_data)
                print(f"[OK] Collected {len(movement_data)} movement(s) for: {station_name}")
            else:
                print(f"[INFO] No movements for station: {station_name}")

        except Exception as e:
            print(f"[EXCEPTION] Error processing station {station_id}: {e}")

        # Delay between stations
        time.sleep(random.uniform(1.0, 2.5))

    # Save collected data
    output_csv_path = workdir / "movements.csv"

    if all_stations_movs:
        df = pd.DataFrame(all_stations_movs)
        df.to_csv(output_csv_path, index=False)
        print(f"[OK] Saved total {len(all_stations_movs)} movements to {output_csv_path}")
    else:
        print("[INFO] No movements collected from any station")


if __name__ == "__main__":
    main()
