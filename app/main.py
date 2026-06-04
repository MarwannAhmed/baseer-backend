import sys
from pathlib import Path

# Allow `from arabic_ocr.X import Y` while the package lives inside app/features/
sys.path.insert(0, str(Path(__file__).resolve().parent / "features"))

from fastapi import FastAPI
from app.routers import analyze

app = FastAPI(title="Baseer Backend")

app.include_router(analyze.router)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}