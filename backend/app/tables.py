from sqlalchemy import Column, Integer, Float, String
from .db import Base

class Station(Base):
    # Static table: one row per station
    __tablename__ = "stations"

    station_id = Column(String, primary_key=True)
    name = Column(String)
    lat = Column(Float)
    lon = Column(Float)
    capacity = Column(Integer)


class Snapshot(Base):
    # Time-series table: one row per station per timestamp
    __tablename__ = "snapshots"

    # Composite primary key means unique per (timestamp, station_id)
    timestamp_utc = Column(String, primary_key=True)
    station_id = Column(String, primary_key=True)

    bikes_available = Column(Integer)
    docks_available = Column(Integer)


class LivePrediction(Base):
    # Latest predictions table (one row per station per timestamp)
    __tablename__ = "live_predictions"

    # latest timestamp + station_id = unique prediction row
    timestamp_utc = Column(String, primary_key=True)
    station_id = Column(String, primary_key=True)

    p_empty_30 = Column(Float)
    p_full_30 = Column(Float)

    bikes_now = Column(Integer)
    docks_now = Column(Integer)