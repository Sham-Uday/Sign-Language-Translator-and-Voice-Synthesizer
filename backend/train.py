import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

DATA_DIR = Path("./second/landmarks") # Adjust if your unzipped folder name differs

X = []
y = []

print("Loading and normalizing pre-extracted landmark coordinates...")

if not DATA_DIR.exists():
    print(f"Error: Directory '{DATA_DIR}' not found. Check your dataset path!")
else:
    for letter_dir in DATA_DIR.iterdir():
        if letter_dir.is_dir() and not letter_dir.name.startswith('.'):
            label = letter_dir.name.upper()
            if len(label) == 1 and label.isalpha():
                for npy_file in letter_dir.glob("*.npy"):
                    try:
                        landmarks = np.load(npy_file)
                        # Ensure shape is (21, 3)
                        if landmarks.shape == (21, 3):
                            pts = landmarks.copy()
                        elif landmarks.size == 63:
                            pts = landmarks.reshape(21, 3)
                        else:
                            continue

                        # --- EXACT SAME NORMALIZATION AS BACKEND ---
                        # 1. Shift relative to wrist (landmark 0)
                        wrist = pts[0]
                        pts = pts - wrist
                        
                        # 2. Scale by maximum distance from wrist
                        max_dist = np.max(np.linalg.norm(pts, axis=1))
                        if max_dist > 0:
                            pts = pts / max_dist
                        
                        features = pts.flatten()
                        # -------------------------------------------

                        if len(features) == 63:
                            X.append(features)
                            y.append(label)
                    except Exception:
                        continue

    X = np.array(X)
    y = np.array(y)

    print(f"Successfully processed and loaded {len(X)} samples!")

    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training Random Forest Classifier on normalized features...")
        model = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Training complete! Model Accuracy: {accuracy * 100:.2f}%")

        joblib.dump(model, "model.pkl")
        print("Saved fresh, normalized model.pkl successfully!")
    else:
        print("Error: No valid landmark samples found. Check your folder structure.")