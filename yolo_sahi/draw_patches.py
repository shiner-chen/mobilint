import cv2
import numpy as np

# Load the image
img_path = 'traffic_jam.jpg'
img = cv2.imread(img_path)

if img is None:
    raise ValueError(f"Could not load image from {img_path}")

# Define the 8 patches with their coordinates
patches = [
    [(0, 0), (640, 0), (640, 640), (0, 640)],  # Top-left
    [(640, 0), (1280, 0), (1280, 640), (640, 640)],  # Top-middle
    [(1280, 0), (1920, 0), (1920, 640), (1280, 640)],  # Top-right
    [(0, 440), (640, 440), (640, 1080), (0, 1080)],  # Bottom-left
    [(640, 440), (1280, 440), (1280, 1080), (640, 1080)],  # Bottom-middle
    [(1280, 440), (1920, 440), (1920, 1080), (1280, 1080)],  # Bottom-right
    [(320, 220), (960, 220), (960, 840), (320, 840)],  # Center-left (overlapping)
    [(960, 220), (1600, 220), (1600, 840), (960, 840)],  # Center-right (overlapping)
]

# Define colors for each patch (BGR format for OpenCV)
colors = [
    (0, 255, 0),      # Green - Top-left
    (255, 0, 0),      # Blue - Top-middle
    (0, 0, 255),      # Red - Top-right
    (255, 255, 0),    # Cyan - Bottom-left
    (255, 0, 255),    # Magenta - Bottom-middle
    (0, 255, 255),    # Yellow - Bottom-right
    (0, 0, 0),        # Black - Center-left
    (255, 255, 255),  # White - Center-right
]

# Draw each patch border
for i, patch in enumerate(patches):
    # Convert patch coordinates to numpy array for cv2.polylines
    pts = np.array(patch, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw the rectangle border
    cv2.polylines(img, [pts], isClosed=True, color=colors[i], thickness=3, lineType=cv2.LINE_AA)
    
    # Optionally add patch number label
    # Get the top-left corner for label placement
    x, y = patch[0]
    cv2.putText(img, f'P{i+1}', (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, colors[i], 2, cv2.LINE_AA)

# Save the result
output_path = 'traffic_jam_with_patches.jpg'
cv2.imwrite(output_path, img)
print(f"Image with patch borders saved to {output_path}")
