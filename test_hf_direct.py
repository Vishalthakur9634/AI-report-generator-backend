from huggingface_hub import InferenceClient
import os

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")
model_id = "openai/whisper-tiny"

client = InferenceClient(token=HF_TOKEN)

# Create a small dummy audio file (it should be valid audio for whisper usually, but let's see if it even reaches the API)
test_file = "tiny_test.wav"
with open(test_file, "wb") as f:
    f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

print(f"Testing HF Client with {model_id}...")
try:
    with open(test_file, "rb") as f:
        audio_data = f.read()
    
    transcription = client.automatic_speech_recognition(
        audio_data,
        model=model_id
    )
    print("Transcription Result:", transcription)
except Exception as e:
    print("HF Client Error:", e)
finally:
    if os.path.exists(test_file):
        os.remove(test_file)
