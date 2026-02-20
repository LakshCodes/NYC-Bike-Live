from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DB_PATH = (BACKEND_DIR / "data" / "bike.db").resolve()

engine = create_engine(f"sqlite:///{DB_PATH}")

df = pd.read_sql("SELECT station_id, timestamp_utc FROM snapshots", engine)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

counts = df.groupby("station_id").size()

print("Stations with any snapshots:", counts.shape[0])
print("Min snapshots per station:", int(counts.min()))
print("Median snapshots per station:", int(counts.median()))
print("90th percentile snapshots/station:", int(counts.quantile(0.9)))
print("Max snapshots per station:", int(counts.max()))

# How many stations have enough for 30min back + 30min forward (>= 31 points gives at least 1 training row)
eligible = (counts >= 31).sum()
print("Stations eligible for 30-min dataset (>=31 snapshots):", int(eligible))