import cv2
import numpy as np
from pathlib import Path

TARGET_LETTER = "O"
DATA_DIR = Path(f"./second/landmarks/{TARGET_LETTER.upper()}")

npy_files = list(DATA_DIR.glob("*.npy"))
if not npy_files:
    print(f"No samples found for letter {TARGET_LETTER}!")
    exit()

print(f"Found {len(npy_files)} samples for '{TARGET_LETTER}'. Press any key to view the next one, or 'q' to quit.")

# MediaPipe Hand Connections mapping for visualization
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),         # Thumb
    (0,5), (5,6), (6,7), (7,8),         # Index
    (5,9), (9,10), (10,11), (11,12),     # Middle
    (9,13), (13,14), (14,15), (15,16),   # Ring
    (13,17), (17,18), (18,19), (19,20),  # Pinky
    (0,17)                              # Palm base
]

for file_path in npy_files:
    # Load the coordinates and ensure shape is (21, 3)
    pts = np.load(file_path)
    if pts.size == 63:
        pts = pts.reshape(21, 3)
    
    # Create a blank black image (500x500 pixels)
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    scaled_pts = []
    for pt in pts:
        # Safely extract x and y
        x = int(float(pt[0]) * 200 + 250)
        y = int(float(pt[1]) * 200 + 250)
        scaled_pts.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1) # Red joints

    # Draw connections
    for connection in HAND_CONNECTIONS:
        pt1 = scaled_pts[connection[0]]
        pt2 = scaled_pts[connection[1]]
        cv2.line(img, pt1, pt2, (0, 255, 0), 2) # Green bones

    cv2.putText(img, f"Letter: {TARGET_LETTER} | {file_path.name}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Landmark Visualizer", img)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()