from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import traceback
from pydantic import BaseModel
import uvicorn
import os
import json
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Radiology Report Generator")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

# Production CORS Configuration (Robust)
origins = ["http://localhost:5173", "http://localhost:3000"]
if FRONTEND_URL and FRONTEND_URL != "*":
    clean_url = FRONTEND_URL.strip().rstrip("/")
    origins.append(clean_url)
    origins.append(clean_url.lower()) # Case-insensitive handle
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True if "*" not in origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "online", "message": "AI Radiology Backend is Running"}

@app.get("/api/test")
async def api_test():
    return {"status": "ok", "message": "The /api/test path works!"}

@app.get("/api/health")
async def api_health():
    return {"status": "ok"}

class ReportRequest(BaseModel):
    transcript: str
    modality: str = "USG" # Default to USG
    
class ReportResponse(BaseModel):
    patientData: dict
    reportText: str
    impression: str

repo_id = "Qwen/Qwen2.5-72B-Instruct"

def extract_json_from_text(text: str) -> dict:
    try:
        md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if md_match:
            return json.loads(md_match.group(1))
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except:
        return None

@app.post("/api/generate_report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    transcript = request.transcript

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HUGGINGFACEHUB_API_TOKEN is missing in the environment.")

    try:
        client = InferenceClient(model=repo_id, token=HF_TOKEN)
        
        modality = request.modality.upper()
        
        # Modality-Specific Prompt Templates (ULTIMATE MEDICAL QUALITY)
        MODALITY_RULES = {
            "USG": "Mandatory Detail: Use 'coarsened echotexture', 'hepatomegaly', 'calculi', 'calyceal dilatation', 'septated collection'. Start every organ with <b>Organ Name</b>. Describe size, echogenicity, morphology, and negatives (e.g., 'no focal lesion seen').",
            "X-RAY": "Describe bone alignment, cortex integrity, joint spaces, lung parenchyma, pleural spaces, and cardiac silhouette. Use 'radiopacity', 'radiolucency', 'costophrenic angles are clear'.",
            "CT": "Detailed Hounsfield Units, contrast enhancement phases, and multi-planar relationship. Use <b>Region</b> format with verbose descriptive anatomy.",
            "MRI": "Signal intensity (T1/T2/FLAIR/STIR/DWI), soft tissue contrast, neural/ligamentous integrity. Detailed sequence-by-sequence analysis.",
            "BLOOD TEST": "Structured table/list. <b>Parameter</b>: Value (Range) [FLAG]. Interpretation of trends.",
            "DOPPLER": "Flow dynamics, RI, PSV, waveforms, arterial/venous patency. Describe spectral broadning or turbulence if applicable."
        }
        
        specific_instructions = MODALITY_RULES.get(modality, "Generate a world-class professional diagnostic report.")

        system_prompt = f"""
        You are a World-Class Senior Radiologist with 20+ years of experience. Your task is to transform raw clinical notes/audio into a DEEP, DETAILED, and HYPER-PROFESSIONAL diagnostic report for: {modality}.
        
        ### 🌟 GOLD STANDARD EXAMPLE (Follow this Level of Detail):
        "The <b>Liver</b> is moderately enlarged in size (~ 18.9 cm) with coarsening of echotexture and shows increased parenchymal echogenicity consistent with grade II / III fatty changes, obscuring the parenchymal details. The intra hepatic biliary passages are not dilated. Portal vein is normal ~ 11.0 mm. No focal lesion is seen."
        "The <b>Gall Bladder</b> is minimally distended (Non-fasting). The CBD appears normal. No evidence of calculi seen."
        "The <b>Impression</b> must be a clinical synthesis: '1. Moderate Hepatomegaly with Grade II / III Fatty changes in liver with coarsening of echotexture. (Adv: LFT and US Guided Elastography correlation).'"

        ### 📋 CORE MANDATES:
        1. **Verbosity**: Do NOT give short answers. Expand every finding into 2-3 detailed sentences.
        2. **Negative Findings**: Explicitly mention what is NORMAL (e.g., "No evidence of focal lesion, calculi, or ductal dilatation is seen").
        3. **Anatomic Precision**: Reference echotexture, echogenicity, outlines, and measurements (~ 00 cm).
        4. **Formatting**: Use `<p>` tags for each section and `<b>` for the Organ/Header.

        ### 📦 JSON RESPONSE (STRICTLY JSON ONLY):
        {{
            "patientData": {{ 
                "patient_name": "Full Name",
                "age": "e.g., 45 Yrs",
                "sex": "Male/Female",
                "ref_doctor": "Dr. Name",
                "date": "Date",
                "uhid": "10XXXXXX",
                "study": "{modality} WHOLE ABDOMEN" (or specific study)
            }},
            "reportText": "<p>The <b>Liver</b> is... [verbose description]</p><p>The <b>Spleen</b> is... [verbose description]</p>",
            "impression": "1. [Significant Finding]\\n2. [Second Finding]"
        }}
        """

        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nRaw Transcript: {transcript}"}
        ]
        
        # Native HuggingFace text-generation call that natively supports chat structures on the free inference API
        response = client.chat_completion(
            messages,
            max_tokens=1024,
            temperature=0.1
        )
        
        result_str = response.choices[0].message.content
        parsed_data = extract_json_from_text(result_str)
        
        if parsed_data and "patientData" in parsed_data:
            return parsed_data
        else:
            print("Failed to parse JSON. Raw output was:", result_str)
            raise ValueError("LLM generated invalid JSON structure.")
            
    except Exception as e:
        print(f"HuggingFace Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HUGGINGFACEHUB_API_TOKEN is missing.")

    try:
        # Read the uploaded file
        audio_data = await file.read()
        
        client = InferenceClient(token=HF_TOKEN)
        
        # Use Distil-Whisper as a potentially free/lighter alternative
        model_id = "distil-whisper/distil-large-v3"
        
        print(f"Transcribing audio with model: {model_id}")
        
        transcription = client.automatic_speech_recognition(
            audio_data,
            model=model_id
        )
        
        return {"text": transcription.text}
        
    except Exception as e:
        print(f"Transcription error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
