from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import joblib
import httpx

from sqlalchemy.orm import Session
from sqlalchemy import func

from .tables import Station, Snapshot, LivePrediction


# -----------------------------
# GBFS endpoints (Citi Bike NYC)
# -----------------------------
STATION_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"


# -----------------------------
# Model paths (local)
# -----------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = (BACKEND_DIR / "models").resolve()

EMPTY_MODEL_PATH = MODEL_DIR / "empty_30_model.joblib"
FULL_MODEL_PATH = MODEL_DIR / "full_30_model.joblib"
THRESH_PATH = MODEL_DIR / "thresholds.joblib"

_models_cache = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(tag: str, msg: str):
    print(f"[{tag}] {utc_now_iso()} {msg}")


# -----------------------------
# Model loader (cached)
# -----------------------------
def load_models():
    """
    Loads models once and caches them for reuse.
    Also loads thresholds.joblib if present.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    if not EMPTY_MODEL_PATH.exists() or not FULL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Expected:\n"
            f"- {EMPTY_MODEL_PATH}\n"
            f"- {FULL_MODEL_PATH}\n"
            "Train and save them first (train_models.py)."
        )

    empty_model = joblib.load(EMPTY_MODEL_PATH)
    full_model = joblib.load(FULL_MODEL_PATH)

    thresholds = {
        "empty_30_threshold": 0.5,
        "full_30_threshold": 0.5,
        # If thresholds.joblib includes a "features" list, we will use it.
        "features": None,
    }

    if THRESH_PATH.exists():
        try:
            t = joblib.load(THRESH_PATH)
            # merge / override
            thresholds.update(t if isinstance(t, dict) else {})
        except Exception as e:
            _log("infer", f"warning: failed to load thresholds.joblib: {e}")

    _models_cache = (empty_model, full_model, thresholds)
    return _models_cache


# -----------------------------
# GBFS fetchers
# -----------------------------
async def fetch_station_information() -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(STATION_INFO_URL)
        r.raise_for_status()
        data = r.json()
        return data["data"]["stations"]


async def fetch_station_status() -> tuple[str, list[dict]]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(STATION_STATUS_URL)
        r.raise_for_status()
        data = r.json()
        stations = data["data"]["stations"]
        # GBFS commonly includes "last_updated" as epoch seconds
        last_updated = data.get("last_updated")
        if isinstance(last_updated, (int, float)):
            ts = datetime.fromtimestamp(last_updated, tz=timezone.utc).replace(microsecond=0).isoformat()
        else:
            ts = utc_now_iso()
        return ts, stations


# -----------------------------
# DB writers
# -----------------------------
async def refresh_stations(db: Session) -> int:
    """
    Pull station_information and upsert into stations table.
    """
    try:
        stations = await fetch_station_information()
    except Exception as e:
        _log("stations", f"fetch failed: {e}")
        return 0

    n = 0
    for s in stations:
        station_id = s.get("station_id")
        if not station_id:
            continue

        obj = Station(
            station_id=str(station_id),
            name=s.get("name"),
            lat=float(s.get("lat")),
            lon=float(s.get("lon")),
        )
        # optional fields if your Station model has them
        if hasattr(obj, "capacity"):
            cap = s.get("capacity")
            obj.capacity = int(cap) if cap is not None else None

        db.merge(obj)
        n += 1

    db.commit()
    _log("stations", f"upserted {n} stations")
    return n


async def poll_status(db: Session) -> int:
    try:
        ts, statuses = await fetch_station_status()
    except Exception as e:
        _log("poll", f"fetch failed: {e}")
        return 0

    # ✅ If feed timestamp didn't advance, don't insert duplicates
    latest_ts = db.query(func.max(Snapshot.timestamp_utc)).scalar()
    if latest_ts == ts:
        _log("poll", f"skipping (timestamp unchanged): {ts}")
        return 0

    n = 0
    for st in statuses:
        station_id = st.get("station_id")
        if not station_id:
            continue

        bikes = st.get("num_bikes_available")
        docks = st.get("num_docks_available")

        snap = Snapshot(
            station_id=str(station_id),
            timestamp_utc=ts,
            bikes_available=int(bikes) if bikes is not None else 0,
            docks_available=int(docks) if docks is not None else 0,
        )

        # ✅ merge = upsert by PK (timestamp_utc + station_id)
        db.merge(snap)
        n += 1

    db.commit()
    _log("poll", f"stored {n} snapshot rows @ {ts}")
    return n


# -----------------------------
# Feature building for inference
# -----------------------------
def _parse_ts(ts_str: str) -> datetime:
    # Handles "2026-02-19T23:12:00+00:00"
    return datetime.fromisoformat(ts_str)


def _get_past_snapshot(db: Session, station_id: str, current_ts: datetime, minutes_back: int) -> Snapshot | None:
    """
    Time-based query (more robust than assuming exact poll intervals).
    Finds the latest snapshot <= current_ts - minutes_back.
    """
    target = current_ts - timedelta(minutes=minutes_back)
    return (
        db.query(Snapshot)
        .filter(Snapshot.station_id == station_id)
        .filter(Snapshot.timestamp_utc <= target.isoformat())
        .order_by(Snapshot.timestamp_utc.desc())
        .first()
    )


def build_feature_rows_for_latest(db: Session) -> tuple[str, list[dict]]:
    """
    Uses the latest snapshot timestamp and builds one feature row per station.
    """
    latest_ts_str = db.query(func.max(Snapshot.timestamp_utc)).scalar()
    if latest_ts_str is None:
        return ("", [])

    latest_ts = _parse_ts(latest_ts_str)

    snaps = db.query(Snapshot).filter(Snapshot.timestamp_utc == latest_ts_str).all()
    if not snaps:
        return (latest_ts_str, [])

    rows: list[dict] = []
    for s in snaps:
        station_id = str(s.station_id)

        bikes_now = int(s.bikes_available)
        docks_now = int(s.docks_available)
        total = bikes_now + docks_now
        pct_bikes = (bikes_now / total) if total else 0.0
        pct_docks = (docks_now / total) if total else 0.0

        # time features
        hour = int(latest_ts.hour)
        dayofweek = int(latest_ts.weekday())  # Monday=0
        is_weekend = 1 if dayofweek >= 5 else 0

        # deltas using history
        s10 = _get_past_snapshot(db, station_id, latest_ts, 10)
        s30 = _get_past_snapshot(db, station_id, latest_ts, 30)

        bikes_delta_10 = bikes_now - int(s10.bikes_available) if s10 else 0
        docks_delta_10 = docks_now - int(s10.docks_available) if s10 else 0

        bikes_delta_30 = bikes_now - int(s30.bikes_available) if s30 else 0
        docks_delta_30 = docks_now - int(s30.docks_available) if s30 else 0

        rows.append({
            "station_id": station_id,
            "timestamp_utc": latest_ts_str,

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
        })

    return (latest_ts_str, rows)


# -----------------------------
# Inference job
# -----------------------------
# ... everything above stays the same ...

def run_inference(db: Session) -> int:
    """
    Predict p(empty_30) and p(full_30) for latest snapshot per station and upsert to predictions table.
    """
    try:
        empty_model, full_model, thr = load_models()
    except Exception as e:
        _log("infer", f"model load failed: {e}")
        return 0

    latest_ts_str, feature_rows = build_feature_rows_for_latest(db)
    if not feature_rows:
        _log("infer", "no feature rows (no snapshots yet?)")
        return 0

    X = pd.DataFrame(feature_rows)

    # If thresholds.joblib included a "features" list, enforce it (and fill missing with 0)
    feat_list = thr.get("features")
    if isinstance(feat_list, list) and len(feat_list) > 0:
        for c in feat_list:
            if c not in X.columns:
                X[c] = 0
        X = X[feat_list].copy()

    # predict probabilities
    p_empty = empty_model.predict_proba(X)[:, 1]
    p_full = full_model.predict_proba(X)[:, 1]

    # Thresholds are optional (useful if you later want to emit binary alerts)
    _ = float(thr.get("empty_30_threshold", 0.5))
    _ = float(thr.get("full_30_threshold", 0.5))

    count = 0
    for i, r in enumerate(feature_rows):
        pe = float(p_empty[i])
        pf = float(p_full[i])

        pred = LivePrediction(
            station_id=r["station_id"],
            timestamp_utc=latest_ts_str,
            p_empty_30=pe,
            p_full_30=pf,
            bikes_now=int(r["bikes_now"]),
            docks_now=int(r["docks_now"]),
        )
        db.merge(pred)
        count += 1

    db.commit()
    _log("infer", f"upserted {count} predictions (ts={latest_ts_str})")
    return count


def run_inference_now(db: Session) -> int:
    return run_inference(db)