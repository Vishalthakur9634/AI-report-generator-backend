from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import get_current_user
from database import templates_collection, reports_collection, fs
from services.ai_service import generate_report_data
from bson import ObjectId
import io
import os
from docxtpl import DocxTemplate

router = APIRouter()

class GenerateReportRequest(BaseModel):
    template_id: str
    transcript: str

@router.post("/generate")
async def generate_report(
    request: GenerateReportRequest,
    current_user: dict = Depends(get_current_user)
):
    # 1. Fetch Template
    template_doc = await templates_collection.find_one({"_id": ObjectId(request.template_id), "center_id": ObjectId(current_user["center_id"])})
    if not template_doc:
        raise HTTPException(status_code=404, detail="Template not found")
        
    detected_tags = template_doc.get("detected_tags", [])
    modality = template_doc.get("modality", "USG")
    
    # 2. Call AI to get structured JSON matching the tags
    try:
        report_data = await generate_report_data(request.transcript, detected_tags, modality)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # 3. Fetch the .docx from GridFS
    try:
        grid_out = await fs.open_download_stream(template_doc["file_id"])
        docx_bytes = await grid_out.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch template file: {str(e)}")
        
    # 4. Inject Data into DOCX using docxtpl
    try:
        doc = DocxTemplate(io.BytesIO(docx_bytes))
        doc.render(report_data)
        
        # Save filled docx to a bytes buffer
        filled_docx_io = io.BytesIO()
        doc.save(filled_docx_io)
        filled_docx_io.seek(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render docx: {str(e)}")
        
    # 5. Save the generated DOCX back to GridFS
    # In production, we'd convert it to PDF here (e.g., using docx2pdf or LibreOffice API)
    # For MVP, we save the filled .docx
    file_name = f"Report_{current_user['full_name']}_{ObjectId()}.docx"
    final_file_id = await fs.upload_from_stream(
        file_name,
        filled_docx_io.read(),
        metadata={"contentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "center_id": current_user["center_id"]}
    )
    
    # 6. Save Report Audit Trail
    report_record = {
        "center_id": ObjectId(current_user["center_id"]),
        "template_id": ObjectId(request.template_id),
        "created_by": ObjectId(current_user["_id"]),
        "status": "completed",
        "file_id": final_file_id,
        "ai_payload": report_data
    }
    result = await reports_collection.insert_one(report_record)
    
    return {
        "message": "Report generated successfully",
        "report_id": str(result.inserted_id),
        "file_id": str(final_file_id),
        "preview_data": report_data
    }

@router.get("/")
async def get_reports(current_user: dict = Depends(get_current_user)):
    cursor = reports_collection.find({"center_id": ObjectId(current_user["center_id"])})
    reports = await cursor.to_list(length=100)
    
    for r in reports:
        r["_id"] = str(r["_id"])
        r["center_id"] = str(r["center_id"])
        r["template_id"] = str(r["template_id"])
        r["created_by"] = str(r["created_by"])
        r["file_id"] = str(r["file_id"])
        
    return reports

from fastapi.responses import StreamingResponse

@router.get("/download/{file_id}")
async def download_report(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Open download stream from GridFS
        grid_out = await fs.open_download_stream(ObjectId(file_id))
        
        # Verify ownership / center_id metadata for security
        file_doc = getattr(grid_out, "file_document", {}) or {}
        metadata = file_doc.get("metadata", {}) or {}
        
        if metadata.get("center_id") and str(metadata["center_id"]) != str(current_user["center_id"]):
            raise HTTPException(status_code=403, detail="Unauthorized access to this file")

        headers = {
            "Content-Disposition": 'attachment; filename="report.docx"',
            "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
        
        async def iterate_file():
            while True:
                chunk = await grid_out.read(1024 * 1024) # Read in 1MB chunks
                if not chunk:
                    break
                yield chunk
                
        return StreamingResponse(iterate_file(), headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

