import warnings
# Suppress scikit-learn parallel joblib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import numpy as np
import joblib

app = FastAPI()

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load your trained model
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Warning: Could not load model.pkl: {e}")

class LandmarkPayload(BaseModel):
    landmarks: Union[List[Dict[str, float]], List[List[float]], List[Any]]

@app.get("/")
async def root():
    return {"status": "online", "message": "ASL Translator API is running"}

@app.post("/predict")
async def predict_sign(payload: LandmarkPayload):
    landmarks = payload.landmarks
    
    if not model or len(landmarks) < 21:
        return {"prediction": "-"}
        
    try:
        pts = []
        for pt in landmarks[:21]:
            if isinstance(pt, dict):
                pts.append([pt.get("x", 0), pt.get("y", 0), pt.get("z", 0)])
            elif isinstance(pt, (list, tuple)) and len(pt) >= 3:
                pts.append([pt[0], pt[1], pt[2]])
                
        if len(pts) == 21:
            pts = np.array(pts)
            wrist = pts[0]
            pts = pts - wrist
            max_dist = np.max(np.linalg.norm(pts, axis=1))
            if max_dist > 0:
                pts = pts / max_dist
            
            features = pts.flatten()
            
            if len(features) == 63:
                X_input = features.reshape(1, -1)
                prediction = model.predict(X_input)[0]
                confidence = np.max(model.predict_proba(X_input))
                
                # Confidence filter
                if confidence > 0.58:
                    return {"prediction": str(prediction)}
                    
    except Exception as e:
        print(f"Prediction error: {e}")
        
    return {"prediction": "-"}