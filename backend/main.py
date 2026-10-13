import json
import warnings
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Warning: model.pkl not found. {e}")

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                landmarks = payload.get("landmarks", [])
                
                result = "-"
                
                if model and len(landmarks) >= 21:
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
                            
                            # Stricter confidence filter to avoid background guesses
                            if confidence > 0.55:
                                result = str(prediction)

                # Safely send, exit loop if connection closed
                await websocket.send_json({"prediction": result})
                
            except json.JSONDecodeError:
                continue
            except Exception:
                break 
                
    except WebSocketDisconnect:
        pass