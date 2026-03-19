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
        ROLE: You are the world's most advanced AI Medical Typist. Your goal is to COMPLETELY replace manual medical typing by transforming brief, shorthand doctor's notes into 5-star, diagnostic-grade medical reports for: {modality}.

        ### 🧠 SHORTHAND EXPANSION ENGINE (Training Samples):
        - INPUT: "liver 18cm, fatty, rest normal"
          OUTPUT: "The <b>Liver</b> is moderately enlarged in size (~ 18.0 cm) with coarsening of echotexture and shows increased parenchymal echogenicity consistent with grade II fatty changes. The intra hepatic biliary passages are not dilated. Portal vein is normal. The <b>Gall Bladder</b>, <b>Pancreas</b>, and <b>Spleen</b> are normal in size and echogenicity with no obvious focal lesions. Both <b>Kidneys</b> are normal in position, outline and echogenicity."
        - INPUT: "xray chest, clear"
          OUTPUT: "The lung fields are clear and well-expanded. No focal consolidations, nodules or masses are seen. Both costophrenic angles are clear. The cardiac silhouette is normal in size and shape. The bony thoracic cage and visualised neck structures are unremarkable."

        ### 📋 TYPIST REPLACEMENT RULES:
        1. **Expansion**: Convert every shorthand word into a full, professional medical description.
        2. **Standard Normals**: If a doctor notes one organ is abnormal but says "rest normal" or doesn't mention others, you MUST provide professional "unremarkable" descriptions for the standard organs of that study.
        3. **Technical Precision**: Use high-level terminology: 'hepatosplenomegaly', 'cholelithiasis', 'nephrolithiasis', 'atelectasis', 'unremarkable'.
        4. **Measurements**: If a measurement is provided (e.g. 12cm), integrate it professionally (~ 12.0 cm).
        5. **Formatting**: Use `<p>` for paragraphs and `<b>` for Organ Names.

        ### 📦 RESPONSE STRUCTURE (JSON ONLY):
        {{
            "patientData": {{ 
                "patient_name": "Name", "age": "Age", "sex": "Sex", "ref_doctor": "Dr. Name", "date": "Date", "uhid": "ID", "study": "FULL STUDY NAME"
            }},
            "reportText": "<p>Professional Findings...</p>",
            "impression": "1. Numerical List of Primary Findings.\\n2. Secondary findings."
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
