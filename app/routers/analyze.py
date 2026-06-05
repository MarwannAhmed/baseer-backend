from fastapi import APIRouter, UploadFile, File, Form, HTTPException, logger
from app.services.command_router import route_command

router = APIRouter()


@router.post("/analyze")
async def analyze(
    command: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    logger.info("Received analyze request with command: %s", command)
    image_bytes = await file.read()

    logger.info("Received file: %s, size: %d bytes", file.filename, len(image_bytes))
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    logger.info("Routing command: %s", command)
    result = route_command(command=command, image_bytes=image_bytes)
    logger.info("Command '%s' processed with result: %s", command, result)
    if (command == 'كشف'): 
        logger.info("Returning object detection result.")
        return {"objects": result}
    logger.info("Returning text extraction result.")
    return {"description": result}
