import requests
import json

url = "http://localhost:8000/api/generate_report"
payload = {
    "transcript": "Patient name Vishal Sharma age 30 male ref doctor Dr Kumar ultrasound whole abdomen. Liver is fatty grade one. Gall bladder is normal. Spleen is normal. Both Kidneys have stones measuring 5mm."
}
headers = {'Content-Type': 'application/json'}

print("Sending request to FastAPI...")
try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Raw Response Text:", response.text)
except Exception as e:
    print("Request Failed:", e)
