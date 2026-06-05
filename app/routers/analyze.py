from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.command_router import route_command

router = APIRouter()


@router.post("/analyze")
async def analyze(
    command: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    result = route_command(command=command, image_bytes=image_bytes)
    if command == "كشف":
        return {"objects": result}
    if command == "لون":
        return {"color": result}
    if command == "نص":
        return {"description": result}
    if command == "نصا":
        return {"description": result}
    if command == "نصعر":
        return {"description": result}
