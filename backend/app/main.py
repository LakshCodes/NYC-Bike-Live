import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import Base, engine, SessionLocal
from .jobs import refresh_stations, poll_status, run_inference_now
from .api import router as api_router

app = FastAPI()
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

async def _poll_loop():
    # poll every 60 seconds forever
    while True:
        db = SessionLocal()
        try:
            await poll_status(db)
            run_inference_now(db)
            print("[loop] polled + inferred")
        except Exception as e:
            print("[loop] error:", repr(e))
        finally:
            db.close()

        await asyncio.sleep(60)

@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        await refresh_stations(db)
        await poll_status(db)
        run_inference_now(db)
        print("[startup] stations refreshed + initial poll + initial inference")
    finally:
        db.close()

    # IMPORTANT: start the background loop
    asyncio.create_task(_poll_loop())
    print("[startup] background poll loop started")