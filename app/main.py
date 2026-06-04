from fastapi import FastAPI
from app.routers import analyze

app = FastAPI(title="Baseer Backend")

app.include_router(analyze.router)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}