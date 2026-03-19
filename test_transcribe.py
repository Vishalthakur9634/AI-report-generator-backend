import requests
import os

url = "http://localhost:8000/api/transcribe"

# Create a dummy audio-like file if none exists for testing
# In a real scenario, you'd use a real .webm or .mp3 file
test_file = "test_audio.webm"
if not os.path.exists(test_file):
    with open(test_file, "wb") as f:
        f.write(b"dummy audio data")

print(f"Testing transcription endpoint with {test_file}...")

try:
    with open(test_file, "rb") as f:
        files = {"file": (test_file, f, "audio/webm")}
        response = requests.post(url, files=files)
    
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("Response JSON:", response.json())
    else:
        print("Error Response:", response.text)
except Exception as e:
    print("Test failed:", e)
finally:
    if os.path.exists(test_file):
        os.remove(test_file)
