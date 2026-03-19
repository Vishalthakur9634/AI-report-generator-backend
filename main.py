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

# Secure CORS Configuration
origins = []
allow_all = False

if FRONTEND_URL and FRONTEND_URL != "*":
    origins.append(FRONTEND_URL.rstrip("/"))
    origins.append("http://localhost:5173")
    origins.append("http://localhost:3000")
else:
    origins = ["*"]
    allow_all = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=not allow_all, # Must be False if origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReportRequest(BaseModel):
    transcript: str
    modality: str = "USG" # Default to USG
    
class ReportResponse(BaseModel):
    patientData: dict
    reportText: str
    impression: str

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
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

@app.get("/")
async def root():
    return {"status": "AI Radiology Report Generator Backend is Running"}

@app.post("/api/generate_report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    transcript = request.transcript

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HUGGINGFACEHUB_API_TOKEN is missing in the environment.")

    try:
        client = InferenceClient(model=repo_id, token=HF_TOKEN)
        
        modality = request.modality.upper()
        
        # Modality-Specific Prompt Templates
        MODALITY_RULES = {
            "USG": "Focus on organ morphology, echotexture (hyperechoic/hypoechoic), and echogenicity. Standard sections: Liver, Gallbladder, Pancreas, Spleen, Kidneys, Bladder, etc.",
            "X-RAY": "Focus on bone alignment, cortex integrity, joint spaces, lung parenchyma, pleural spaces, and cardiac silhouette. Use terminology like 'radiopacity' and 'radiolucency'.",
            "CT": "Focus on Hounsfield Units (density), contrast enhancement patterns, and multi-planar anatomy. Standard sections: Head/Chest/Abdomen/Pelvis anatomy as per study.",
            "MRI": "Focus on signal intensity (T1/T2/FLAIR/DWI), soft tissue detail, and ligamentous/neural integrity. Standard sections: Sequences performed, findings per sequence/region.",
            "BLOOD TEST": "Focus on laboratory parameters (CBC, LFT, KFT, etc.). Format as a table or structured list with Reference Ranges if provided. Flag abnormal values as High/Low.",
            "DOPPLER": "Focus on vascular flow dynamics, resistivity indices (RI), peak systolic velocities (PSV), and spectral waveforms. Mention arterial/venous patency."
        }
        
        specific_instructions = MODALITY_RULES.get(modality, "Generate a professional radiology report as per standard protocols.")

        system_prompt = f"""
        You are a highly specialized AI Radiology and Pathology assistant. Your core mission is to transform raw clinical dictation into professional, diagnostic-grade medical reports for the modality: {modality}.

        ### MODALITY-SPECIFIC FOCUS:
        {specific_instructions}

        ### DOCUMENT STRUCTURE & STYLE:
        1. **Tone**: Use formal, objective, and precise medical terminology.
        2. **Organ/System Organization**: findings should be grouped logically.
        3. **Expansion**: Convert shorthand (e.g., 'Lung clear') into professional text (e.g., 'The lung fields are clear and well-expanded. No focal consolidations or pleural effusions are seen.').

        ### TECHNICAL CONSTRAINTS:
        - **NEVER** hallucinate findings.
        - **Formatting**: Use HTML for `reportText`. Use `<b>` for organ/test names and `<p>` for paragraphs.
        - **Impression**: Concise, numbered list.

        ### JSON OUTPUT SCHEMA:
        Respond ONLY with a JSON object in this exact format:
        {{
            "patientData": {{ 
                "patient_name": "Full Name",
                "age": "Age",
                "sex": "Male/Female",
                "ref_doctor": "Referring Physician",
                "date": "Report Date",
                "uhid": "UHID (if found)",
                "study": "Full Name of Study (e.g., {modality} WHOLE ABDOMEN)"
            }},
            "reportText": "<p><b>ORGAN/TEST:</b> Findings...</p>",
            "impression": "1. Main finding.\\n2. Secondary finding."
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
