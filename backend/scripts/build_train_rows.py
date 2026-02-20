from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DB_PATH = (BACKEND_DIR / "data" / "bike.db").resolve()

OUT_DIR = (BACKEND_DIR.parent / "data_processed").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = (OUT_DIR / "train_rows.parquet").resolve()

print("SCRIPT_DIR =", SCRIPT_DIR)
print("BACKEND_DIR =", BACKEND_DIR)
print("DB_PATH =", DB_PATH)
print("DB exists?", DB_PATH.exists())
print("OUT_PATH =", OUT_PATH)

engine = create_engine(f"sqlite:///{DB_PATH}")

print("Loading snapshots...")
df = pd.read_sql("SELECT * FROM snapshots", engine)
print(f"Loaded {len(df)} rows")

if len(df) == 0:
    raise SystemExit("No snapshot data in DB yet. Keep polling first.")

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
df = df.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

# ---- LABEL THRESHOLDS (tune these) ----
# More learnable than exact 0; still useful operationally.
EMPTY_THRESHOLD_BIKES = 1     # near-empty within 30 min
FULL_THRESHOLD_DOCKS = 1      # near-full within 30 min

# ---- TIME STEP ASSUMPTION ----
# Your poll interval seems to be ~2 minutes.
# 15 steps ~ 30 minutes
PAST_STEPS_30 = 15
PAST_STEPS_10 = 5
FUTURE_STEPS_30 = 15

rows = []
print("Building features...")

for station_id, g in df.groupby("station_id"):
    g = g.sort_values("timestamp_utc").reset_index(drop=True)

    # Need enough history + enough future
    if len(g) < (PAST_STEPS_30 + FUTURE_STEPS_30 + 1):
        continue

    for i in range(len(g)):
        if i < PAST_STEPS_30 or i + FUTURE_STEPS_30 >= len(g):
            continue

        current = g.iloc[i]
        future = g.iloc[i + FUTURE_STEPS_30]

        bikes_now = int(current["bikes_available"])
        docks_now = int(current["docks_available"])

        # 30 min deltas
        bikes_30min_ago = int(g.iloc[i - PAST_STEPS_30]["bikes_available"])
        docks_30min_ago = int(g.iloc[i - PAST_STEPS_30]["docks_available"])
        bikes_delta_30 = bikes_now - bikes_30min_ago
        docks_delta_30 = docks_now - docks_30min_ago

        # 10 min deltas
        bikes_10min_ago = int(g.iloc[i - PAST_STEPS_10]["bikes_available"])
        docks_10min_ago = int(g.iloc[i - PAST_STEPS_10]["docks_available"])
        bikes_delta_10 = bikes_now - bikes_10min_ago
        docks_delta_10 = docks_now - docks_10min_ago

        # ratios (better signal than raw counts)
        total_now = bikes_now + docks_now
        pct_bikes = (bikes_now / total_now) if total_now else 0.0
        pct_docks = (docks_now / total_now) if total_now else 0.0

        ts = current["timestamp_utc"]
        hour = int(ts.hour)
        dayofweek = int(ts.dayofweek)  # Mon=0
        is_weekend = 1 if dayofweek >= 5 else 0

        # Labels (NEAR empty/full)
        empty_30 = 1 if int(future["bikes_available"]) <= EMPTY_THRESHOLD_BIKES else 0
        full_30  = 1 if int(future["docks_available"]) <= FULL_THRESHOLD_DOCKS else 0

        rows.append({
            "station_id": str(station_id),
            "timestamp_utc": ts.isoformat(),

            "bikes_now": bikes_now,
            "docks_now": docks_now,

            "bikes_delta_10min": bikes_delta_10,
            "docks_delta_10min": docks_delta_10,

            "bikes_delta_30min": bikes_delta_30,
            "docks_delta_30min": docks_delta_30,

            "pct_bikes": pct_bikes,
            "pct_docks": pct_docks,

            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,

            "empty_30": empty_30,
            "full_30": full_30,
        })

train_df = pd.DataFrame(rows)
print(f"Created {len(train_df)} training rows")

if len(train_df) == 0:
    print("No training rows created. Keep polling longer.")
    train_df.to_parquet(OUT_PATH, index=False)
    raise SystemExit(0)

print("empty_30 rate:", train_df["empty_30"].mean())
print("full_30 rate :", train_df["full_30"].mean())

train_df.to_parquet(OUT_PATH, index=False)
print("Saved to", OUT_PATH)