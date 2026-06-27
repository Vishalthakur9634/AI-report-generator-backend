from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List
from auth import get_current_user
from database import templates_collection, fs
from bson import ObjectId
import re

router = APIRouter()

@router.post("/upload")
async def upload_template(
    name: str = Form(...),
    modality: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are allowed.")
    
    # Read file content to save to GridFS
    content = await file.read()
    
    # Store in GridFS
    file_id = await fs.upload_from_stream(
        file.filename,
        content,
        metadata={"contentType": file.content_type, "center_id": current_user["center_id"]}
    )
    
    # In a real scenario, we might want to parse the .docx here using python-docx
    # to automatically extract all {{tags}} for validation.
    # For MVP, we assume standard tags: patient_name, age, sex, findings, impression
    detected_tags = ["patient_name", "age", "sex", "findings", "impression"]
    
    template_doc = {
        "center_id": ObjectId(current_user["center_id"]),
        "name": name,
        "modality": modality,
        "file_id": file_id,
        "detected_tags": detected_tags,
        "uploaded_by": ObjectId(current_user["_id"])
    }
    
    result = await templates_collection.insert_one(template_doc)
    
    return {
        "message": "Template uploaded successfully",
        "template_id": str(result.inserted_id),
        "file_id": str(file_id)
    }

@router.get("/")
async def get_templates(current_user: dict = Depends(get_current_user)):
    cursor = templates_collection.find({"center_id": ObjectId(current_user["center_id"])})
    templates = await cursor.to_list(length=100)
    
    # Format ObjectIds for JSON serialization
    for t in templates:
        t["_id"] = str(t["_id"])
        t["center_id"] = str(t["center_id"])
        t["file_id"] = str(t["file_id"])
        t["uploaded_by"] = str(t["uploaded_by"])
        
    return templates
