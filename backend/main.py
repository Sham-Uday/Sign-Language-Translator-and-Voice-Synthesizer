from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "FastAPI Sign Language Backend is running!"}

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Frontend connected via WebSocket")
    
    try:
        while True:
            # Receive the JSON payload containing the 21 landmarks from the frontend
            data = await websocket.receive_text()
            payload = json.loads(data)
            landmarks = payload.get("landmarks", [])
            
            # TODO: Pass `landmarks` into your trained Random Forest / LSTM model here!
            # For now, let's return a dummy prediction to test the loop
            prediction_result = "A" if len(landmarks) > 0 else "No hand detected"

            # Send the prediction back to the React frontend
            await websocket.send_json({"prediction": prediction_result})
            
    except WebSocketDisconnect:
        print("Frontend disconnected")
    except Exception as e:
        print(f"Error: {e}")