import cv2
import numpy as np
import time

def check_camera_access():
    """Check if we can access the camera with a test frame"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    
    # Try to read a frame
    ret, _ = cap.read()
    cap.release()
    return ret

def request_camera_permission():
    """Guide user to grant camera permissions"""
    print("\n⚠️ Camera access required ⚠️")
    print("Please grant camera permissions:")
    print("1. Open System Settings → Privacy & Security")
    print("2. Select Camera")
    print("3. Enable access for your terminal/Python app")
    print("4. Try again after granting permissions\n")
    time.sleep(2)  # Give time to read the message

def live_undertone_detection():
    # Initialize camera with retries
    max_retries = 3
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            if attempt == 0:
                request_camera_permission()
            print(f"Camera initialization attempt {attempt + 1}/{max_retries}")
            time.sleep(2)
            continue
            
        print("Camera successfully initialized!")
        cv2.namedWindow("Live Undertone Detection", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture error")
                break
            
            # Analysis and display
            display_frame = frame.copy()
            undertone = analyze_undertone_frame(frame)
            
            cv2.putText(display_frame, f"Undertone: {undertone}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 's' to save, 'q' to quit", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Live Undertone Detection", display_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("undertone_snapshot.jpg", frame)
                print("Snapshot saved!")
        
        cap.release()
        cv2.destroyAllWindows()
        return
    
    print("❌ Failed to access camera after multiple attempts")
    print("Possible solutions:")
    print("- Check if another app is using the camera")
    print("- Restart your computer")
    print("- Try a different camera if available")

def analyze_undertone_frame(frame):
    """Analyze undertone from a camera frame"""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhanced skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 255, 255]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        skin_pixels = lab[mask > 0]
        if len(skin_pixels) == 0:
            return "NO SKIN"
        
        mean_a = np.mean(skin_pixels[:, 1])
        mean_b = np.mean(skin_pixels[:, 2])
        diff = mean_b - mean_a
        
        if diff < -3: return "WARM"
        elif diff > 10: return "COOL"
        elif diff > 5: return "WARM" if mean_a > 130 else "COOL"
        return "NEUTRAL"
    except Exception as e:
        print(f"Analysis error: {e}")
        return "ERROR"

if __name__ == "__main__":
    # First check if we have permission
    if not check_camera_access():
        request_camera_permission()
    
    # Then start the detection
    live_undertone_detection()
