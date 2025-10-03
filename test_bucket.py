import os
import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate(os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"])
firebase_admin.initialize_app(cred, {
    "storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"]
})
bucket = storage.bucket()
blobs = list(bucket.list_blobs(prefix="noseprints/"))
print(f"Found {len(blobs)} blobs:")
for b in blobs:
    print(b.name)
