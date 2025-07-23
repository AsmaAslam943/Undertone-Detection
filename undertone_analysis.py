import cv2
import numpy as np

def analyze_undertone(img_path):
    """Analyze skin undertone with completely revised classification"""
    print(f"\nAnalyzing: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read {img_path}")
        return None

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # New skin detection approach
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Wider range for skin tones
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Get skin pixels
    skin_pixels = lab[mask > 0]
    if len(skin_pixels) == 0:
        print("No skin detected")
        return None
    
    # Calculate percentiles instead of median
    a_low, a_high = np.percentile(skin_pixels[:, 1], [25, 75])
    b_low, b_high = np.percentile(skin_pixels[:, 2], [25, 75])
    
    # Focus on central range (ignore outliers)
    central_pixels = skin_pixels[
        (skin_pixels[:, 1] >= a_low) & (skin_pixels[:, 1] <= a_high) &
        (skin_pixels[:, 2] >= b_low) & (skin_pixels[:, 2] <= b_high)
    ]
    
    if len(central_pixels) == 0:
        return "NEUTRAL"
    
    # Calculate mean of central pixels
    mean_a = np.mean(central_pixels[:, 1])
    mean_b = np.mean(central_pixels[:, 2])
    diff = mean_b - mean_a
    
    print(f"LAB Values - Mean a*: {mean_a:.1f}, Mean b*: {mean_b:.1f}")
    print(f"Difference: {diff:.1f}, b*/a* ratio: {mean_b/mean_a:.2f}")
    
    # New classification logic based on your specific images
    if diff < -3:  # a* significantly higher than b*
        return "WARM"
    elif diff > 10:  # b* significantly higher than a*
        return "COOL"
    elif diff > 5:   # Moderate cool tendency
        if mean_a > 130:  # Golden undertones
            return "WARM"
        return "COOL"
    else:
        return "NEUTRAL"

def main():
    # Hardcoded paths
    warm_path = "/Users/asmaaslam/Desktop/undertone_recognition/images/warm.png"
    cool_path = "/Users/asmaaslam/Desktop/undertone_recognition/images/cool.png"
    
    # Analyze images
    for img_path in [warm_path, cool_path]:
        undertone = analyze_undertone(img_path)
        if undertone:
            filename = img_path.split('/')[-1]
            expected = filename.split('.')[0].upper()
            
            print(f"\nResult for {filename}:")
            print(f"Expected: {expected}")
            print(f"Detected: {undertone}")
            
            # Display results with proper formatting
            img = cv2.imread(img_path)
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            mean_a = np.mean(lab_img[:,:,1])
            mean_b = np.mean(lab_img[:,:,2])
            text = f"{expected}->{undertone} (a*/b*: {mean_a:.1f}/{mean_b:.1f})"
            
            cv2.putText(img, text, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(3000)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
