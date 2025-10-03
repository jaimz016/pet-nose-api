# save_as: prepare_dataset.py
import os, json
from firebase_admin import credentials, storage, initialize_app
import cv2, numpy as np

# load credentials JSON file (local file on your PC)
cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
initialize_app(cred, {"storageBucket":"pawtag-6cf73.appspot.com"})
bucket = storage.bucket()

def extract_nose_roi_auto(img_bgr, target_size=224, margin=0.2):
    # copy your function here (same as server.py)
    ...

os.makedirs("data", exist_ok=True)
blobs = list(bucket.list_blobs(prefix="noseprints/"))
for blob in blobs:
    if not blob.name.lower().endswith((".jpg",".jpeg",".png")): continue
    pet_id = blob.name.split("/")[1]
    outdir = os.path.join("data", pet_id)
    os.makedirs(outdir, exist_ok=True)
    content = blob.download_as_bytes()
    img_bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    roi = extract_nose_roi_auto(img_bgr)
    fname = os.path.join(outdir, os.path.basename(blob.name))
    cv2.imwrite(fname, roi)
    print("Saved", fname)
