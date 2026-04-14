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
        
        # Modality-Specific Rules (Medical Typist Replacement Level)
        MODALITY_RULES = {
            "USG": "Focus: Morphology, echotexture, echogenicity, organ sizes. Use: 'coarsened', 'hepatomegaly', 'calculi', 'calyceal dilatation'.",
            "X-RAY": "Focus: Bone alignment, lung parenchyma, pleural spaces, cardiac silhouette. Use: 'radiopacity', 'radiolucency', 'costophrenic angles clear'.",
            "CT/MRI": "Focus: Density (HU), Signal Intensity (T1/T2), anatomical relationships, contrast phases. Use: 'unremarkable', 'no focal enhancement'.",
            "BLOOD TEST": "Focus: Table format. Parameter: Value (Range). Use: 'Clinically significant elevation', 'Within biological reference range'.",
            "DOPPLER": "Focus: RI, PSV, flow dynamics, waveforms. Use: 'Monophasic/Triphasic flow', 'No significant stenosis'."
        }
        
        specific_instructions = MODALITY_RULES.get(modality, "Generate a comprehensive diagnostic report.")

        system_prompt = f"""
        ROLE: You are the world's most advanced AI Medical Radiologist and Typist. Your goal is to transform brief, shorthand doctor's notes into 5-star, diagnostic-grade, professional medical reports for: {{modality}}.

        ### 📋 MODALITY-SPECIFIC MEDICAL RULES:
        - **USG (Ultrasound)**: Focus on echotexture, echogenicity, organ sizes, and margins. Use terms like 'hepatomegaly', 'cholelithiasis', 'unremarkable parenchymal echotexture'.
        - **X-RAY**: Focus on bone alignment, lung parenchyma, pleural spaces, and cardiac silhouette. Use 'radiopacity', 'radiolucency', 'costophrenic angles clear', 'well-expanded lung fields'.
        - **CT SCAN**: Focus on density (Hounsfield Units), enhancement patterns, and anatomical relationships. Use 'hyper/isodense', 'no focal enhancement', 'unremarkable windowing'.
        - **MRI SCAN**: Focus on signal intensity (T1/T2/FLAIR), diffusion, and anatomical precision. Use 'hyperintense', 'hypointense', 'no restricted diffusion'.
        - **DOPPLER**: Focus on flow dynamics, RI, PSV, and waveforms. Use 'monophasic/triphasic flow', 'no significant stenosis', 'normal spectral waveform'.

        ### 🧠 SHORTHAND EXPANSION ENGINE:
        - Convert shorthand words into full, professional medical sentences.
        - **Standard Normals**: If a doctor notes one organ is abnormal but says "rest normal" or doesn't mention others, you MUST provide professional "unremarkable" descriptions for the standard organs of that specific study.
        - **Precision**: Integrate measurements precisely (~ 12.0 cm). Use `<b>` tags for Organ Names or Landmarks.
        - **Format**: Use `<p>` for findings and numbered list for impressions.

        ### 🗳️ RESPONSE STRUCTURE (JSON ONLY):
        {{
            "patientData": {{ 
                "patient_name": "Name", "age": "Age", "sex": "Sex", "ref_doctor": "Dr. Name", "date": "Date", "uhid": "P-ID", "study": "FULL STUDY NAME IN CAPS"
            }},
            "reportText": "<p>Professional Findings with medical precision...</p>",
            "impression": "1. Numerical List of Primary Findings.\n2. Secondary findings."
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
