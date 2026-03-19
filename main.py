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
        
        # Modality-Specific Prompt Templates (Refined for Professional Quality)
        MODALITY_RULES = {
            "USG": "Mimic professional radiology style: Use terms like 'coarsened echotexture', 'hepatomegaly', 'calculi', 'calyceal dilatation', 'septated collection'. Always start organ findings with the <b>Organ Name</b> in bold. Include dimensions (~ 0.0 cm) if mentioned. Standard sections: Liver, Gallbladder, Pancreas, Spleen, Kidneys, Bladder, etc.",
            "X-RAY": "Focus on bone alignment, cortex integrity, joint spaces, lung parenchyma, pleural spaces, and cardiac silhouette. Use terminology like 'radiopacity' and 'radiolucency'. Use <b>Organ/Region</b> format.",
            "CT": "Focus on Hounsfield Units (density), contrast enhancement patterns, and multi-planar anatomy. Describe specific phase enhancement if applicable. Use <b>Region</b> format.",
            "MRI": "Focus on signal intensity (T1/T2/FLAIR/DWI), soft tissue detail, and neural integrity. Use <b>Sequence/Region</b> format.",
            "BLOOD TEST": "Focus on parameters (CBC, LFT, KFT). Format as a structured list with <b>Parameter</b>: Value (Range). Highlight abnormal values.",
            "DOPPLER": "Focus on flow dynamics, resistivity indices (RI), PSV, and spectral waveforms. Mention arterial/venous patency. Use <b>Vessel</b> format."
        }
        
        specific_instructions = MODALITY_RULES.get(modality, "Generate a professional diagnostic report.")

        system_prompt = f"""
        You are a Senior Radiologist AI assistant. Your task is to convert raw clinical notes or audio transcripts into a formal, diagnostic-grade medical report for: {modality}.
        
        REFERENCE STYLE (Mimic this EXACTLY):
        - Style: "The <b>Liver</b> is moderately enlarged in size (~ 18.9 cm) with coarsening of echotexture..."
        - Detail: Mention specific grades (e.g., Grade II/III fatty changes), measurements, and 'No evidence of' for negatives.
        - Clarity: Ensure findings are distinct and descriptive.
        
        ### MODALITY-SPECIFIC RULES:
        {specific_instructions}

        ### DOCUMENT STRUCTURE:
        1. **Tone**: Objective, formal, and authoritative.
        2. **Formatting**: Use HTML for `reportText`. 
           - Start each finding with: `<p>The <b>Organ Name</b> is ...</p>` or `<b>Organ Name</b>: ...`
           - Use `<p>` tags for each distinct section.
        3. **Expansion**: Expand shorthand (e.g., 'liver big' -> 'The <b>Liver</b> is moderately enlarged in size, showing features of hepatomegaly.').
        4. **Impression**: Mandatory concise, numbered list of the most significant findings. Include clinical recommendations (Adv:) if appropriate.

        ### JSON RESPONSE SCHEMA (Respond ONLY with JSON):
        {{
            "patientData": {{ 
                "patient_name": "Full Name",
                "age": "e.g., 36 Yrs",
                "sex": "Male/Female",
                "ref_doctor": "Referring Physician Name",
                "date": "Date of Report",
                "uhid": "UHID number",
                "study": "{modality} WHOLE ABDOMEN" (or specific study mentioned)
            }},
            "reportText": "<p>The <b>Liver</b> is...</p><p>The <b>Gall Bladder</b> is...</p>",
            "impression": "1. Main Diagnostic Finding.\\n2. Secondary Finding."
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
