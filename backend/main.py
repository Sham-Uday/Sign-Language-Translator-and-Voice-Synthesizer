import json
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
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
            payload = json.loads(data)
            
            landmarks = payload.get("landmarks", [])
            
            # Default to empty if no hand landmarks are present
            result = ""
            
            if model and len(landmarks) >= 21:
                features = []
                for pt in landmarks[:21]:
                    if isinstance(pt, dict):
                        features.extend([pt.get("x", 0), pt.get("y", 0), pt.get("z", 0)])
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 3:
                        features.extend([pt[0], pt[1], pt[2]])
                
                if len(features) == 63:
                    X_input = np.array(features).reshape(1, -1)
                    prediction = model.predict(X_input)[0]
                    confidence = np.max(model.predict_proba(X_input))
                    
                    # Only accept prediction if confidence is high enough
                    if confidence > 0.6:
                        result = str(prediction)

            await websocket.send_json({"prediction": result})
            
    except WebSocketDisconnect:
        print("Frontend disconnected")