import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState("Waiting for signs...");
  const [currentWord, setCurrentWord] = useState("");
  const wsRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const lastAddedCharRef = useRef("");

  // 1. Initialize WebSocket connection to FastAPI
  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws/predict");

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const pred = data.prediction;
      setPrediction(pred);

      // Basic logic to append stable letters to the word box
      if (pred && pred.length === 1 && pred !== lastAddedCharRef.current && pred !== "...") {
        setCurrentWord((prev) => prev + pred);
        lastAddedCharRef.current = pred;
      }
    };

    wsRef.current.onerror = (err) => {
      console.error("WebSocket error:", err);
    };

    return () => {
      wsRef.current?.close();
    };
  }, []);

  // 2. Initialize MediaPipe HandLandmarker
  useEffect(() => {
    async function setupHandLandmarker() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });

        startWebcam();
      } catch (error) {
        console.error("Error setting up MediaPipe:", error);
      }
    }

    setupHandLandmarker();
  }, []);

  // 3. Start Webcam Stream
  const startWebcam = () => {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            predictWebcam();
          };
        }
      })
      .catch((err) => console.error("Webcam access error:", err));
  };

  // Text-to-Speech Handler
  const speakWord = () => {
    if ('speechSynthesis' in window && currentWord) {
      const utterance = new SpeechSynthesisUtterance(currentWord);
      window.speechSynthesis.speak(utterance);
    }
  };

  // Frame-by-frame loop with crash protection
  let lastVideoTime = -1;
  const predictWebcam = async () => {
    try {
      if (videoRef.current && handLandmarkerRef.current && videoRef.current.readyState >= 2) {
        if (videoRef.current.currentTime !== lastVideoTime) {
          lastVideoTime = videoRef.current.currentTime;
          
          const startTimeMs = performance.now();
          const results = handLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);

          const canvasElement = canvasRef.current;
          const videoElement = videoRef.current;

          if (canvasElement && videoElement) {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;

            const canvasCtx = canvasElement.getContext("2d");
            if (canvasCtx) {
              canvasCtx.save();
              canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

              const drawingUtils = new DrawingUtils(canvasCtx);

              if (results.landmarks && results.landmarks.length > 0) {
                for (const landmarks of results.landmarks) {
                  drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                    color: "#22c55e",
                    lineWidth: 4
                  });
                  drawingUtils.drawLandmarks(landmarks, {
                    color: "#ef4444",
                    lineWidth: 2,
                    radius: 5
                  });
                }

                const landmarks = results.landmarks[0];
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ landmarks }));
                }
              } else {
                // Reset tracker memory when hand leaves the screen
                lastAddedCharRef.current = "";
              }
              canvasCtx.restore();
            }
          }
        }
      }
    } catch (error) {
      console.error("Error in prediction loop:", error);
    }

    requestAnimationFrame(predictWebcam);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-slate-950 text-white p-6">
      <h1 className="text-3xl font-bold mb-6">Real-Time Sign Language Translator</h1>
      
      <Card className="w-full max-w-2xl bg-slate-900 border-slate-800 p-4 shadow-xl space-y-4">
        <CardContent className="flex flex-col items-center space-y-4 p-0">
          <div className="relative w-full aspect-[4/3] bg-black rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
            <video 
              ref={videoRef} 
              playsInline 
              muted 
              className="absolute inset-0 w-full h-full -scale-x-100 object-contain" 
            />
            <canvas 
              ref={canvasRef} 
              className="absolute inset-0 w-full h-full -scale-x-100 pointer-events-none object-contain" 
            />
          </div>
          
          <div className="flex items-center space-x-2 w-full justify-between pt-2">
            <span className="text-sm text-slate-400">Current Letter:</span>
            <Badge variant="outline" className="text-lg px-4 py-1 text-green-400 border-green-500/50">
              {prediction}
            </Badge>
          </div>

          {/* Word Accumulation & Speech Controls */}
          <div className="flex flex-col w-full space-y-2 bg-slate-950 p-4 rounded-lg border border-slate-800">
            <div className="flex justify-between items-center">
              <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Translated Transcript</span>
              <div className="space-x-2">
                <Button variant="outline" size="sm" onClick={() => setCurrentWord("")} className="border-slate-700 text-slate-300 hover:bg-slate-800">
                  Clear
                </Button>
                <Button size="sm" onClick={speakWord} className="bg-green-600 hover:bg-green-700 text-white">
                  Speak Word
                </Button>
              </div>
            </div>
            <div className="min-h-[40px] text-xl font-medium tracking-wide text-green-300 flex items-center">
              {currentWord || <span className="text-slate-600 italic text-sm">Spelled words will appear here...</span>}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}