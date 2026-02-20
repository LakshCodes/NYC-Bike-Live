from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# adjust path if needed
DB_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "bike.db"
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")

df = pd.read_sql("SELECT * FROM snapshots", engine)
out = Path(__file__).resolve().parents[2] / "data_processed"
out.mkdir(parents=True, exist_ok=True)

df.to_parquet(out / "snapshots.parquet", index=False)
print("Wrote:", out / "snapshots.parquet", "rows:", len(df))