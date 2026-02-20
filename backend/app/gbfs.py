import httpx

STATION_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

HEADERS = {"User-Agent": "nyc-bike-live/1.0 (student project)"}

def fetch_station_information():
    with httpx.Client(timeout=20, headers=HEADERS) as client:
        r = client.get(STATION_INFO_URL)
        r.raise_for_status()
        j = r.json()
    return j["data"]["stations"]

def fetch_station_status():
    with httpx.Client(timeout=20, headers=HEADERS) as client:
        r = client.get(STATION_STATUS_URL)
        r.raise_for_status()
        j = r.json()
    return j["data"]["stations"]
