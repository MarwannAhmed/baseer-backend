from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.command_router import route_command

router = APIRouter()


@router.post("/analyze")
async def analyze(
    command: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    print(f"Received command: {command}, file: {file.filename}")
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")
    
    print(f"Image file size: {len(image_bytes)} bytes")
    result = route_command(command=command, image_bytes=image_bytes)
    if (command == 'كشف'): 
        print(f"Detection result: {result}")
        return {"objects": result}
    print(f"Extraction result: {result}")
    return {"description": result}
