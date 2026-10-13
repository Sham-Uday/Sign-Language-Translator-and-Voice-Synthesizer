import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Point to the unzipped landmark folders (A, B, C...)
DATA_DIR = Path("./second/landmarks") # Adjust if there's an inner folder

X = []
y = []

print("Loading pre-extracted landmark coordinates...")

for letter_dir in DATA_DIR.iterdir():
    if letter_dir.is_dir() and not letter_dir.name.startswith('.'):
        label = letter_dir.name.upper()
        if len(label) == 1 and label.isalpha():
            for npy_file in letter_dir.glob("*.npy"):
                try:
                    landmarks = np.load(npy_file)
                    # Flatten (21, 3) matrix into a 63-element feature vector
                    features = landmarks.flatten()
                    if len(features) == 63:
                        X.append(features)
                        y.append(label)
                except Exception:
                    continue

X = np.array(X)
y = np.array(y)

print(f"Successfully loaded {len(X)} pre-processed samples!")

if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Training complete! Model Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, "model.pkl")
    print("Saved real model.pkl successfully!")
else:
    print("Error: No .npy files found. Check your folder path.")