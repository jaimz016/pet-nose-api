from google.cloud import storage
import os
import requests

# --- 1️⃣ Set environment variable for Firebase credentials ---
os.environ['FIREBASE_SERVICE_ACCOUNT_JSON'] = r"C:\Users\james edwin frongoso\Desktop\pawtag-backend\serviceAccountKey.json"

# --- 2️⃣ Create a Firebase Storage client ---
client = storage.Client.from_service_account_json(
    os.environ['FIREBASE_SERVICE_ACCOUNT_JSON']
)

# --- 3️⃣ Get your bucket ---
bucket = client.get_bucket("pawtag-6cf73.firebasestorage.app")


# --- 4️⃣ Specify the image blob path in Firebase ---
blob_path = "noseprints/fw4EHEFHFR339GBZXPWk/1.jpg"
blob = bucket.blob(blob_path)

# --- 5️⃣ Download the image locally ---
local_filename = "1.jpg"
blob.download_to_filename(local_filename)
print(f"Downloaded {blob_path} as {local_filename}")

# --- 6️⃣ Send the image to the FastAPI /identify endpoint ---
url = "http://127.0.0.1:8000/identify"
with open(local_filename, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# --- 7️⃣ Print the API response ---
print("API response:", response.json())
