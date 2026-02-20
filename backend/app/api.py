from fastapi import APIRouter
from sqlalchemy import desc
from sqlalchemy import text
from .jobs import run_inference_now

from .db import SessionLocal
from .tables import Station, Snapshot, LivePrediction

router = APIRouter()

@router.get("/station_count")
def station_count():
    db = SessionLocal()
    try:
        return {"stations": db.query(Station).count()}
    finally:
        db.close()

@router.post("/run_inference_now")
def run_inference_now_endpoint():
    db = SessionLocal()
    try:
        n = run_inference_now(db)
        return {"ok": True, "predictions_upserted": int(n)}
    finally:
        db.close()

@router.get("/timestamp_count")
def timestamp_count():
    db = SessionLocal()
    try:
        n = db.execute(text("SELECT COUNT(DISTINCT timestamp_utc) FROM snapshots")).scalar()
        return {"distinct_snapshot_timestamps": int(n or 0)}
    finally:
        db.close()

@router.get("/snapshot_count")
def snapshot_count():
    db = SessionLocal()
    try:
        return {"snapshots": db.query(Snapshot).count()}
    finally:
        db.close()

@router.get("/last_poll")
def last_poll():
    db = SessionLocal()
    try:
        row = db.query(Snapshot).order_by(desc(Snapshot.timestamp_utc)).first()
        return {"last_timestamp_utc": row.timestamp_utc if row else None}
    finally:
        db.close()

@router.get("/prediction_count")
def prediction_count():
    db = SessionLocal()
    try:
        return {"predictions": db.query(LivePrediction).count()}
    finally:
        db.close()

@router.get("/live.geojson")
def live_geojson():
    db = SessionLocal()
    try:
        latest = db.query(LivePrediction).order_by(desc(LivePrediction.timestamp_utc)).first()
        if not latest:
            return {"type": "FeatureCollection", "features": []}

        latest_ts = latest.timestamp_utc

        preds = db.query(LivePrediction).filter(LivePrediction.timestamp_utc == latest_ts).all()
        stations = db.query(Station).all()
        station_map = {s.station_id: s for s in stations}

        features = []
        for p in preds:
            s = station_map.get(p.station_id)
            if not s or s.lat is None or s.lon is None:
                continue

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(s.lon), float(s.lat)]},
                "properties": {
                    "station_id": p.station_id,
                    "name": s.name,
                    "timestamp_utc": latest_ts,
                    "bikes_now": p.bikes_now,
                    "docks_now": p.docks_now,
                    "p_empty_30": p.p_empty_30,
                    "p_full_30": p.p_full_30,
                }
            })

        return {"type": "FeatureCollection", "features": features}
    finally:
        db.close()