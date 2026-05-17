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

# Explicit CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    findings: list[dict] # List of { "organ": "...", "description": "..." }
    impression: str


repo_id = "Qwen/Qwen2.5-72B-Instruct"

def extract_json_from_text(text: str) -> dict:
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
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
        ROLE: You are an Elite Medical Typist with 25+ years of experience in Radiology and Pathology centers (e.g., JP Diagnostics). Your expertise lies in transforming brief, shorthand doctor's notes into professional, legally-defensible, and diagnostic-grade medical reports for: {{modality}}.

        ### 📋 ELITE CLINICAL GUIDELINES:
        - **Terminology**: Use high-fidelity terms: 'parenchymal echotexture', 'costophrenic angles', 'no focal consolidations', 'unremarkable windowing', 'restricted diffusion'.
        - **Standard of Care**: For any study where an organ isn't mentioned, provide a "Standard Normal" description as per institutional protocols.
        - **Structure**: Every report MUST be subdivided into organs/regions. Output MUST be an array of finding objects.
        - **Measurements**: Integrate measurements professionally as (~ X.X cm).
        - **Impression**: Summarize only clinical significances in a numbered list. Use 'Clinical Correlation Recommended' if findings are ambiguous.

        ### 🧠 MODALITY-SPECIFIC REPLACEMENT RULES:
        - **USG**: Focus on Morphology, echo-patterns, and borders.
        - **X-RAY**: Focus on Alignment, Parenchyma, and Cardiac silhouette.
        - **CT/MRI**: Focus on Signal Intensity, Enhancement, and Volume.
        - **DOPPLER**: Focus on RI, PSV, Waveforms (Triphasic/Biphasic).

        ### 🗳️ RESPONSE STRUCTURE (STRICT JSON ONLY):
        OUTPUT EXACTLY AND ONLY VALID JSON. DO NOT WRAP IN MARKDOWN OR CODE BLOCKS. NO PREAMBLE. NO POSTAMBLE.
        {{
            "patientData": {{ 
                "patient_name": "Name", 
                "age": "Age", 
                "sex": "Sex", 
                "ref_doctor": "Dr. Name", 
                "uhid": "P101000...", 
                "study": "FULL STUDY NAME (e.g. USG WHOLE ABDOMEN MALE)"
            }},
            "findings": [
                {{ "organ": "LIVER", "description": "The Liver is moderately enlarged in size (~ 18.9 cm) with coarsening of echotexture..." }},
                {{ "organ": "GALL BLADDER", "description": "The Gall Bladder is minimally distended (Non-fasting). The CBD appears normal." }},
                ...
            ],
            "impression": "1. Numerical List of Primary Findings.\n2. Potential clinical implications."
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
        error_msg = str(e)
        print(f"HuggingFace Processing error: {error_msg}")
        if "401" in error_msg or "unauthorized" in error_msg.lower() or "invalid token" in error_msg.lower():
            raise HTTPException(status_code=401, detail="HuggingFace Token Expired or Invalid. Please update your API Token.")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {error_msg}")

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
