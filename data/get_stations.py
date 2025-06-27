import requests
import pandas as pd

base_url = "https://wiediversistmeingarten.org/api"
stations_url = f"{base_url}/station"

response = requests.get(stations_url)
stations_data = response.json()

station_list = []

for station in stations_data:
    last_movement = station.get("lastMovement")
    station_list.append({
        "station_id": station["station_id"],
        "name": station["name"],
        # "lat": station["location"]["lat"],
        # "lng": station["location"]["lng"],
        # "sensebox_id": station.get("sensebox_id", "")
        "last_movement_date": last_movement.get("createdAt") if last_movement else None,
    })

stations_df = pd.DataFrame(station_list)
stations_df.to_csv("data/station_ids.csv", index=False)
print("Saved to station_ids.csv")