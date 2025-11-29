import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, storage
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import tempfile

print("🔧 Starting application initialization...")

# ---------------------------
# Firebase Init (Fixed)
# ---------------------------
def init_firebase():
    print("🚀 Initializing Firebase...")
    
    # Get the JSON content from environment variable
    firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS")
    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET")

    print(f"📝 Firebase creds available: {firebase_creds_json is not None}")
    print(f"📦 Bucket name available: {bucket_name is not None}")
    
    if bucket_name:
        print(f"📦 Bucket name: {bucket_name}")
    
    if not firebase_creds_json:
        raise RuntimeError("FIREBASE_CREDENTIALS environment variable not set")
    if not bucket_name:
        raise RuntimeError("FIREBASE_STORAGE_BUCKET environment variable not set")

    try:
        print("🔍 Parsing Firebase credentials JSON...")
        # Parse the JSON to validate it
        creds_dict = json.loads(firebase_creds_json)
        print("✅ JSON parsing successful")
        
        print("📄 Creating temporary credential file...")
        # Create a temporary file with proper naming
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(creds_dict, temp_file)
            temp_path = temp_file.name
        print(f"📄 Temporary file created at: {temp_path}")
        
        print("🔑 Initializing Firebase credentials...")
        # Initialize Firebase with the temp file path
        cred = credentials.Certificate(temp_path)
        print("✅ Firebase credentials created")
        
        print("🔥 Initializing Firebase app...")
        firebase_admin.initialize_app(cred, {
            "storageBucket": bucket_name
        })
        print("✅ Firebase app initialized")
        
        print("🪣 Getting storage bucket...")
        # Get the bucket reference
        bucket = storage.bucket()
        print(f"✅ Storage bucket obtained: {bucket.name}")
        
        print("🧹 Cleaning up temporary file...")
        # Clean up the temp file
        os.unlink(temp_path)
        print("✅ Temporary file cleaned up")
        
        print("🎉 Firebase initialized successfully!")
        return bucket
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        raise RuntimeError(f"Invalid Firebase credentials JSON: {e}")
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"📋 Stack trace: {traceback.format_exc()}")
        raise RuntimeError(f"Firebase initialization failed: {e}")

# Initialize Firebase
print("🔄 Attempting Firebase initialization...")
try:
    bucket = init_firebase()
    print("✅ Firebase initialization completed successfully")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    # Don't raise the exception here, let the app start so we can see other errors
    bucket = None

# ---------------------------
# ORB setup
# ---------------------------
print("🎯 Setting up ORB detector...")
orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8)
print("✅ ORB detector ready")

# ---------------------------
# Helper Functions
# ---------------------------
def extract_nose_roi_auto(img_bgr, target_size=224, margin=0.2):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w//2, h//2

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_c, best_dist = None, 1e9
    for c in contours:
        x, y, wc, hc = cv2.boundingRect(c)
        ccx, ccy = x+wc//2, y+hc//2
        area = wc*hc
        dist = np.hypot(ccx-cx, ccy-cy)/(area+1e-6)
        if area > 500 and dist < best_dist:
            best_dist = dist
            best_c = (x,y,wc,hc)

    if best_c is None:
        roi = gray
    else:
        x, y, wc, hc = best_c
        dx, dy = int(wc*margin), int(hc*margin)
        x1 = max(0, x-dx); y1 = max(0, y-dy)
        x2 = min(w, x+wc+dx); y2 = min(h, y+hc+dy)
        roi = gray[y1:y2, x1:x2]

    resized = cv2.resize(roi, (target_size, target_size))
    return resized

def apply_CLAHE(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def extract_orb(gray):
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des

def match_orb(des1, des2, kp1, kp2, ratio=0.75):
    if des1 is None or des2 is None:
        return 0

    # Create BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter out invalid match pairs
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    if not good:
        return 0

    # Compute score as ratio of good matches to total keypoints
    score = len(good) / max(len(kp1), len(kp2))
    return score


# ---------------------------
# Data Augmentation (Improved)
# ---------------------------
def augment_image(img):
    """Return multiple augmented versions of the same image (rotation, flip, zoom, brightness, blur, noise)."""
    aug_list = []

    # Rotation
    for angle in [-15, 0, 15]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        aug_list.append(rotated)

    # Brightness variations
    brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
    aug_list.extend([brighter, darker])

    # Blur
    blurred = cv2.GaussianBlur(img, (5,5), 1)
    aug_list.append(blurred)

    # Horizontal Flip
    flipped = cv2.flip(img, 1)
    aug_list.append(flipped)

    # Slight Zoom (crop and resize back)
    h, w = img.shape
    crop = img[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]
    zoomed = cv2.resize(crop, (w, h))
    aug_list.append(zoomed)

    # Add Gaussian Noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    aug_list.append(noisy)

    return aug_list


# ---------------------------
# Database from Firebase
# ---------------------------
db = {}

def build_db():
    global db
    print("🏗️ Building database from Firebase Storage...")
    
    if bucket is None:
        print("❌ Cannot build DB - Firebase bucket is not available")
        return
        
    try:
        print("📡 Listing blobs from Firebase Storage...")
        blobs = list(bucket.list_blobs(prefix="noseprints/"))
        print(f"📁 Found {len(blobs)} blobs total")
        
        image_count = 0
        for blob in blobs:
            if not blob.name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
                
            image_count += 1
            print(f"🖼️ Processing image {image_count}: {blob.name}")
            
            pet_id = blob.name.split('/')[1]
            print(f"🐾 Extracted pet ID: {pet_id}")
            
            content = blob.download_as_bytes()
            img_bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            print(f"📐 Image dimensions: {img_bgr.shape}")

            roi = extract_nose_roi_auto(img_bgr)
            preproc = apply_CLAHE(roi)
            kp, des = extract_orb(preproc)
            print(f"🔑 Found {len(kp) if kp else 0} keypoints")

            if pet_id not in db:
                db[pet_id] = []

            # Original
            db[pet_id].append({"des": des, "kp": kp, "path": blob.name})

            # Augmented versions
            aug_images = augment_image(preproc)
            print(f"🔄 Generated {len(aug_images)} augmented versions")
            for aug in aug_images:
                kp_a, des_a = extract_orb(aug)
                db[pet_id].append({"des": des_a, "kp": kp_a, "path": blob.name + "_aug"})

        print(f"✅ DB build complete. Processed {image_count} images")
        
    except Exception as e:
        print(f"❌ Error building database: {e}")
        import traceback
        print(f"📋 Stack trace: {traceback.format_exc()}")

print("🔄 Building database...")
build_db()

# ---------------------------
# FastAPI App
# ---------------------------
print("🚀 Creating FastAPI app...")
app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://yourfrontenddomain.com",  # add if deployed
    "*",  # fallback
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/identify")
async def identify(file: UploadFile = File(...), min_score: float = 0.33):
    """
    Identify pet nose print from uploaded image.
    min_score: minimum matching score (0-1) to consider a valid match.
    Default is 0.6 (60%).
    """
    print(f"🔍 Identification request received. File: {file.filename}, Min score: {min_score}")
    
    content = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    print(f"📐 Uploaded image dimensions: {img_bgr.shape}")
    
    roi = extract_nose_roi_auto(img_bgr)
    preproc = apply_CLAHE(roi)
    kp_q, des_q = extract_orb(preproc)
    print(f"🔑 Query image keypoints: {len(kp_q) if kp_q else 0}")

    scores = []
    print(f"🔎 Searching through {len(db)} pets in database...")
    for pet_id, entries in db.items():
        best = 0
        for e in entries:
            des_db = e["des"]
            kp_db = e["kp"]
            s = match_orb(des_q, des_db, kp_q, kp_db)
            best = max(best, s)
        scores.append({"pet_id": pet_id, "score": best, "score_percent": f"{best*100:.2f}%"})
        print(f"🐾 Pet {pet_id}: best score = {best:.4f}")

    # Sort by raw score for comparison
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    # Check if best match meets minimum threshold
    if not scores or scores[0]["score"] < min_score:
        print("❌ No match found above threshold")
        return {
            "success": False,
            "message": "No matching pet found in database",
            "best_score": scores[0]["score_percent"] if scores else "0.00%",
            "threshold": f"{min_score*100:.0f}%"
        }
    
    # Filter out matches below threshold and return top 3
    valid_matches = [
        {"pet_id": s["pet_id"], "score": s["score_percent"]} 
        for s in scores if s["score"] >= min_score
    ]
    
    print(f"✅ Found {len(valid_matches)} valid matches")
    return {
        "success": True,
        "message": f"Found {len(valid_matches)} matching pet(s)",
        "matches": valid_matches[:3]
    }

# ---------------------------
# Inspect DB contents (runs once at startup)
# ---------------------------
print("📊 Inspecting DB contents...")
total_entries = 0
for pet_id, entries in db.items():
    print(f"\n🐾 Pet ID: {pet_id}")
    print(f"📁 Total images (including augmented): {len(entries)}")
    for e in entries:
        print(f" - {e['path']}")
    total_entries += len(entries)
print(f"\n📈 Total entries in DB (all pets + augmented): {total_entries}")

# ---------------------------
# Health check endpoint
# ---------------------------
@app.get("/")
async def root():
    return {
        "message": "PawTag Backend API", 
        "status": "running",
        "database_loaded": len(db) > 0,
        "pets_in_db": len(db),
        "firebase_ready": bucket is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "firebase_initialized": bucket is not None,
        "database_entries": sum(len(entries) for entries in db.values()),
        "pets_in_database": len(db)
    }

# ---------------------------
# Evaluation Endpoint
# ---------------------------
@app.get("/evaluate")
async def evaluate(threshold: float = 0.3):
    print(f"📊 Evaluation request with threshold: {threshold}")
    genuine_scores, impostor_scores = [], []
    pet_ids = list(db.keys())

    for i, pid in enumerate(pet_ids):
        entries = db[pid]
        for e in entries:
            des_q, kp_q = e["des"], e["kp"]
            if des_q is None:
                continue

            # Genuine comparisons
            for e2 in entries:
                if e["path"] == e2["path"]:
                    continue
                s = match_orb(des_q, e2["des"], kp_q, e2["kp"])
                genuine_scores.append(s)

            # Impostor comparisons
            for j, pid2 in enumerate(pet_ids):
                if pid == pid2:
                    continue
                for e2 in db[pid2]:
                    s = match_orb(des_q, e2["des"], kp_q, e2["kp"])
                    impostor_scores.append(s)

    # Convert to binary predictions
    y_true = np.concatenate([
        np.ones(len(genuine_scores)),  # genuine = 1
        np.zeros(len(impostor_scores)) # impostor = 0
    ])
    y_pred = np.concatenate([
        np.array(genuine_scores) >= threshold,
        np.array(impostor_scores) >= threshold
    ])

    # --- Optional: Find best threshold automatically ---
    best_acc, best_th = 0, 0
    for t in np.linspace(0.05, 0.5, 10):
        y_pred_t = np.concatenate([
            np.array(genuine_scores) >= t,
            np.array(impostor_scores) >= t
        ])
        acc_t = accuracy_score(y_true, y_pred_t)
        if acc_t > best_acc:
            best_acc, best_th = acc_t, t
    print(f"🎯 Best Accuracy: {best_acc:.4f} at threshold {best_th:.2f}")
    # -----------------------------------------------------

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # FMR / FNMR / EER
    fmr = np.mean([s >= threshold for s in impostor_scores])
    fnmr = np.mean([s < threshold for s in genuine_scores])

    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, np.concatenate([genuine_scores, impostor_scores]))
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

    print("✅ Evaluation completed")
    return {
        "Confusion_Matrix": cm.tolist(),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1_Score": float(f1),
        "FMR": float(fmr),
        "FNMR": float(fnmr),
        "EER": float(eer)
    }

print("🎉 Application startup complete!")
print("📡 Server is ready to handle requests")