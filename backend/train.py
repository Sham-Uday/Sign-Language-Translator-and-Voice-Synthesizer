import time
import numpy as np
from pathlib import Path
import joblib
import warnings

# Suppress sklearn and lightgbm parallel/convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

DATA_DIR = Path("./second/landmarks")
X = []
y = []

print("Loading landmark dataset from ./second/...")
if not DATA_DIR.exists():
    print(f"Error: {DATA_DIR} directory not found! Please check your path.")
    exit()

for letter_dir in DATA_DIR.iterdir():
    if letter_dir.is_dir():
        letter = letter_dir.name
        for npy_file in letter_dir.glob("*.npy"):
            data = np.load(npy_file)
            if data.size == 63:
                X.append(data.flatten())
                y.append(letter)

X = np.array(X)
y = np.array(y)
print(f"Successfully loaded {len(X)} samples across {len(np.unique(y))} classes.")

if len(X) == 0:
    print("Error: No samples found. Check your dataset directory path.")
    exit()

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Support Vector Classifier": SVC(kernel='rbf', C=1.0),
    "LightGBM": LGBMClassifier(n_estimators=150, random_state=42, verbose=-1, n_jobs=-1),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    # Calculate requested metrics in order (using Macro average for multi-class fairness)
    acc = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    misclass_rate = (1.0 - (acc / 100.0)) * 100
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Misclass": misclass_rate,
        "Recall": recall,
        "F1": f1,
        "Time": train_time
    })
    
    # Save Random Forest as default production model
    if name == "Random Forest":
        joblib.dump(model, "model.pkl")

# Print clean comparison summary table in exact order
print("\n" + "="*95)
print(f"{'MODEL PERFORMANCE METRICS COMPARISON':^95}")
print("="*95)
print(f"{'Model Name':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Misclass.':<9} | {'Recall':<8} | {'F1 Score':<8} | {'Time':<6}")
print("-" * 95)
for r in results:
    print(f"{r['Model']:<22} | {r['Accuracy']:>6.2f}%  | {r['Precision']:>6.2f}%   | {r['Misclass']:>6.2f}%   | {r['Recall']:>6.2f}%  | {r['F1']:>6.2f}%  | {r['Time']:>4.2f}s")
print("="*95)
print("\nRandom Forest model successfully saved as model.pkl!")