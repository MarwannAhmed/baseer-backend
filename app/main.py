from fastapi import FastAPI

app = FastAPI(title="Baseer Backend")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/hello")
def hello() -> dict:
    return {"message": "Hello, World!"}