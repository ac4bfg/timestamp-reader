from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict, Any
from app.services.ocr_service import process_image_from_bytes

router = APIRouter()

@router.post("/extract-watermark", response_model=Dict[str, Any])
async def extract_watermark(file: UploadFile = File(...)):
    """
    Extract timestamp and GPS coordinates from uploaded photo watermark.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        result = process_image_from_bytes(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
