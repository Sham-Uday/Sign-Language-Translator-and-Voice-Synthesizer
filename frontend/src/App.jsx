import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const lastAddedCharRef = useRef("");
  const missingFramesCountRef = useRef(0);

  const [prediction, setPrediction] = useState("-");
  const [currentWord, setCurrentWord] = useState("");
  const [wsStatus, setWsStatus] = useState("Disconnected");

  // 1. Initialize WebSocket Connection
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/predict");
    wsRef.current = ws;

    ws.onopen = () => setWsStatus("Connected");
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const pred = data.prediction;
        setPrediction(pred || "-");

        if (pred && pred.length === 1 && pred !== lastAddedCharRef.current && pred !== "-" && pred !== "...") {
          setCurrentWord((prev) => prev + pred);
          lastAddedCharRef.current = pred;
        }
      } catch (err) {
        console.error("WebSocket message parsing error:", err);
      }
    };
    ws.onerror = () => setWsStatus("Error");
    ws.onclose = () => setWsStatus("Disconnected");

    return () => ws.close();
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
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
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

  const speakWord = () => {
    if ('speechSynthesis' in window && currentWord.trim()) {
      const utterance = new SpeechSynthesisUtterance(currentWord);
      window.speechSynthesis.speak(utterance);
    }
  };

  // 4. Detection Loop
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
                // Hand is present, reset missing frame tracking
                missingFramesCountRef.current = 0;

                for (const landmarks of results.landmarks) {
                  drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                    color: "#22c55e",
                    lineWidth: 3
                  });
                  drawingUtils.drawLandmarks(landmarks, {
                    color: "#ef4444",
                    lineWidth: 1.5,
                    radius: 4
                  });
                }

                const landmarks = results.landmarks[0];
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ landmarks }));
                }
              } else {
                // Hand is missing, count frames before resetting active letter state
                missingFramesCountRef.current += 1;
                if (missingFramesCountRef.current > 10) { // ~1/3rd of a second threshold
                  lastAddedCharRef.current = "";
                  setPrediction("-");
                }
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
    <div style={styles.appContainer}>
      {/* Header Bar */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={{
            ...styles.dot,
            backgroundColor: wsStatus === "Connected" ? "#10b981" : "#f43f5e"
          }} />
          <h1 style={styles.title}>ASL Real-Time Translator</h1>
        </div>
        <div style={styles.headerRight}>
          <span style={styles.badge}>WS: {wsStatus}</span>
          <span style={styles.badge}>MediaPipe + FastAPI</span>
        </div>
      </header>

      {/* Main Single-Screen Grid */}
      <main style={styles.mainGrid}>
        
        {/* Left Video Area */}
        <div style={styles.videoCard}>
          <div style={styles.cardHeader}>
            <span style={styles.cardHeaderTitle}>Live Camera Feed</span>
            <span style={styles.activeTag}>CAMERA ACTIVE</span>
          </div>

          <div style={styles.videoWrapper}>
            <video 
              ref={videoRef} 
              playsInline 
              muted 
              style={styles.video} 
            />
            <canvas 
              ref={canvasRef} 
              style={styles.canvas} 
            />

            <div style={styles.overlayBadge}>
              <span style={styles.overlayText}>Detected:</span>
              <span style={styles.overlayLetter}>{prediction}</span>
            </div>
          </div>
        </div>

        {/* Right Sidebar Area */}
        <div style={styles.sidebar}>
          
          {/* Active Letter */}
          <div style={styles.letterCard}>
            <span style={styles.sectionTitle}>Active Sign Letter</span>
            <div style={styles.letterDisplay}>{prediction}</div>
          </div>

          {/* Transcript & Controls */}
          <div style={styles.transcriptCard}>
            <div style={styles.transcriptSection}>
              <div style={styles.transcriptHeader}>
                <span style={styles.sectionTitle}>Translated Transcript</span>
                <span style={styles.charCount}>{currentWord.length} chars</span>
              </div>

              <div style={styles.transcriptBox}>
                {currentWord || <span style={styles.placeholder}>Spelled signs will accumulate here...</span>}
              </div>
            </div>

            <div style={styles.buttonGroup}>
              <button 
                onClick={() => setCurrentWord((prev) => prev + " ")}
                style={styles.btnSecondary}
              >
                Space
              </button>
              <button 
                onClick={() => { setCurrentWord(""); lastAddedCharRef.current = ""; }} 
                style={styles.btnSecondary}
              >
                Clear
              </button>
              <button 
                onClick={speakWord} 
                style={styles.btnPrimary}
              >
                🔊 Speak
              </button>
            </div>
          </div>

        </div>

      </main>
    </div>
  );
}

const styles = {
  appContainer: {
    height: "100vh",
    width: "100vw",
    backgroundColor: "#020617",
    color: "#f8fafc",
    fontFamily: "system-ui, -apple-system, sans-serif",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    boxSizing: "border-box",
  },
  header: {
    borderBottom: "1px solid #1e293b",
    backgroundColor: "#0f172a",
    padding: "8px 16px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    flexShrink: 0,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  dot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
  },
  title: {
    fontSize: "14px",
    fontWeight: "700",
    margin: 0,
  },
  headerRight: {
    display: "flex",
    gap: "8px",
  },
  badge: {
    fontSize: "10px",
    fontFamily: "monospace",
    backgroundColor: "#1e293b",
    color: "#94a3b8",
    padding: "2px 8px",
    borderRadius: "4px",
    border: "1px solid #334155",
  },
  mainGrid: {
    flex: 1,
    padding: "12px",
    display: "grid",
    gridTemplateColumns: "2fr 1fr",
    gap: "12px",
    minHeight: 0,
    overflow: "hidden",
    boxSizing: "border-box",
  },
  videoCard: {
    backgroundColor: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "8px",
    padding: "8px",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    overflow: "hidden",
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    paddingBottom: "6px",
    borderBottom: "1px solid #1e293b",
    marginBottom: "8px",
    flexShrink: 0,
  },
  cardHeaderTitle: {
    fontSize: "12px",
    color: "#94a3b8",
  },
  activeTag: {
    fontSize: "10px",
    backgroundColor: "rgba(16, 185, 129, 0.1)",
    color: "#34d399",
    border: "1px solid rgba(16, 185, 129, 0.2)",
    padding: "1px 6px",
    borderRadius: "4px",
    fontWeight: "600",
  },
  videoWrapper: {
    position: "relative",
    width: "100%",
    height: "100%",
    backgroundColor: "#000000",
    borderRadius: "6px",
    overflow: "hidden",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  video: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "cover",
    transform: "scaleX(-1)",
  },
  canvas: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "cover",
    transform: "scaleX(-1)",
    pointerEvents: "none",
  },
  overlayBadge: {
    position: "absolute",
    top: "10px",
    right: "10px",
    backgroundColor: "rgba(2, 6, 23, 0.85)",
    border: "1px solid #1e293b",
    borderRadius: "6px",
    padding: "4px 10px",
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  overlayText: {
    fontSize: "10px",
    color: "#94a3b8",
    fontWeight: "600",
  },
  overlayLetter: {
    fontSize: "18px",
    fontWeight: "900",
    color: "#34d399",
  },
  sidebar: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    minHeight: 0,
    overflow: "hidden",
  },
  letterCard: {
    backgroundColor: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "8px",
    padding: "12px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  sectionTitle: {
    fontSize: "10px",
    fontWeight: "600",
    color: "#94a3b8",
    textTransform: "uppercase",
    marginBottom: "6px",
  },
  letterDisplay: {
    height: "56px",
    width: "56px",
    borderRadius: "8px",
    backgroundColor: "#020617",
    border: "1px solid #1e293b",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "28px",
    fontWeight: "900",
    color: "#34d399",
  },
  transcriptCard: {
    flex: 1,
    backgroundColor: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "8px",
    padding: "12px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
    minHeight: 0,
    overflow: "hidden",
  },
  transcriptSection: {
    display: "flex",
    flexDirection: "column",
    flex: 1,
    minHeight: 0,
  },
  transcriptHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "6px",
    flexShrink: 0,
  },
  charCount: {
    fontSize: "10px",
    color: "#64748b",
  },
  transcriptBox: {
    flex: 1,
    padding: "10px",
    backgroundColor: "#020617",
    borderRadius: "6px",
    border: "1px solid #1e293b",
    fontSize: "14px",
    color: "#f8fafc",
    wordBreak: "break-word",
    overflowY: "auto",
    minHeight: 0,
  },
  placeholder: {
    color: "#475569",
    fontSize: "12px",
    fontStyle: "italic",
  },
  buttonGroup: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: "6px",
    marginTop: "10px",
    flexShrink: 0,
  },
  btnSecondary: {
    backgroundColor: "#1e293b",
    color: "#cbd5e1",
    border: "1px solid #334155",
    padding: "6px",
    borderRadius: "6px",
    fontSize: "11px",
    fontWeight: "600",
    cursor: "pointer",
  },
  btnPrimary: {
    backgroundColor: "#059669",
    color: "#ffffff",
    border: "none",
    padding: "6px",
    borderRadius: "6px",
    fontSize: "11px",
    fontWeight: "600",
    cursor: "pointer",
  },
};