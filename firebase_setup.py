import firebase_admin
from firebase_admin import credentials, db

# Path to your service account key JSON file
SERVICE_ACCOUNT_KEY_PATH = "argovet-firebase-adminsdk-bjghw-ecfddcaee6.json"

# Your Firebase Database URL
DATABASE_URL = "https://argovet-default-rtdb.firebaseio.com"

try:
    # Initialize Firebase app only if it is not already initialized
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred, {
            'databaseURL': DATABASE_URL
        })
    else:
        print("Firebase app already initialized.")

except Exception as e:
    raise ValueError(f"Failed to initialize Firebase: {e}")
