import cv2

# Set your video path
video = r"Subject 2 - female GO Vehicle OFT.mp4"
cap = cv2.VideoCapture(video)
ret, frame = cap.read()

if ret:
    print("--- SELECT YOUR ARENA ---")
    print("Drag a box around the mouse area. Press ENTER when done.")
    # This opens a window; draw the box
    roi = cv2.selectROI("Arena_Selector", frame)
    cv2.destroyAllWindows()
    
    # OpenCV gives (x, y, w, h). 
    # DLC/SuperAnimal often wants [x1, x2, y1, y2]
    x, y, w, h = roi
    coords = [x, x + w, y, y + h]
    
    print(f"\nYour DLC-ready ROI is: {coords}")
else:
    print("Error: Could not read video.")
cap.release()