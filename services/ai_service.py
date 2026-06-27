import os
import json
import re
import asyncio
from huggingface_hub import InferenceClient

MODEL = "Qwen/Qwen2.5-72B-Instruct"

def extract_json(text: str) -> dict:
    try:
        # Try to find JSON block in markdown code fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            return json.loads(fence_match.group(1).strip())

        # Try to find raw JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
    except Exception as e:
        print(f"JSON parse error: {e}")
    return None

async def generate_report_data(transcript: str, detected_tags: list, modality: str) -> dict:
    """
    Calls Hugging Face Serverless Inference API to process the medical transcript
    and outputs a strict JSON matching the detected_tags from the center's DOCX template.
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(model=MODEL, token=token)
    
    # Build schema example
    schema_example = {tag: f"<value for {tag}>" for tag in detected_tags}
    
    system_prompt = f"""You are an Elite Medical Typist with 25+ years of experience in Radiology and Pathology.
Transform the raw doctor's notes into a professional medical report.

MODALITY: {modality}

CRITICAL RULES:
- Use proper clinical terminology (e.g. "coarsened echotexture", "costophrenic angles clear").
- If an organ or field is not mentioned, provide a standard normal description.
- For "findings": write detailed paragraph(s) for each organ/system. Use newlines between organs.
- For "impression": write a numbered list of key clinical findings.
- OUTPUT ONLY RAW JSON. No markdown, no explanation, no preamble.

REQUIRED JSON SCHEMA (output exactly these keys):
{json.dumps(schema_example, indent=2)}"""

    user_message = f"Raw Transcript:\n{transcript}"

    try:
        loop = asyncio.get_event_loop()
        
        # run inference in thread executor to avoid blocking fastapi main event loop
        def call_hf():
            return client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
        response = await loop.run_in_executor(None, call_hf)
        raw_output = response.choices[0].message.content
        print(f"HuggingFace raw output: {raw_output}")
        
        parsed = extract_json(raw_output)
        if not parsed:
            raise ValueError("Failed to parse JSON from AI response")
            
        return parsed
        
    except Exception as e:
        print(f"HuggingFace AI Error: {e}")
        raise ValueError(f"AI generation failed: {str(e)}")
