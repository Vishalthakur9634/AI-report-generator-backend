from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import traceback
import uvicorn
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from database import init_db
from routes import auth_routes, template_routes, report_routes

# Try importing OpenAI for transcription
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key and openai_key != "dummy_key":
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_key)
    except ImportError:
        client = None
else:
    client = None

app = FastAPI(title="AI Radiology SaaS Backend")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Initializing Database...")
    await init_db()

# Mount Routers
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])
app.include_router(template_routes.router, prefix="/api/templates", tags=["templates"])
app.include_router(report_routes.router, prefix="/api/reports", tags=["reports"])

@app.get("/")
async def health_check():
    return {"status": "online", "message": "AI Radiology SaaS Backend is Running"}

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. If OpenAI key is available, use OpenAI Whisper
    if client:
        try:
            temp_file_path = f"temp_{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                buffer.write(await file.read())
                
            with open(temp_file_path, "rb") as audio_file:
                transcription = await client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            os.remove(temp_file_path)
            return {"text": transcription.text}
        except Exception as e:
            print(f"OpenAI Transcription error: {e}")
            traceback.print_exc()
            print("Falling back to Hugging Face...")

    # 2. Free tier fallback: use Hugging Face Whisper
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        hf_client = InferenceClient(token=token)
        
        file_bytes = await file.read()
        
        def call_hf_transcribe():
            return hf_client.audio_to_text(
                file_bytes,
                model="openai/whisper-large-v3"
            )
            
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, call_hf_transcribe)
        return {"text": text}
        
    except Exception as e:
        print(f"Hugging Face Transcription error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
