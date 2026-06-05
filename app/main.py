from fastapi import FastAPI

app = FastAPI(title="Baseer Backend")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
