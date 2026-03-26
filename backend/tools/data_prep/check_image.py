import cv2
import numpy as np

# Read the image as-is, including potential 16-bit depth
img = cv2.imread("dataset/images/00001_930831_fa_a.ppm", cv2.IMREAD_UNCHANGED)

if img is None:
    print("Check your file path.")
else:
    # If the image is 16-bit (Type 2 or 18 in OpenCV), scale it to 8-bit for viewing
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    
    # Convert BGR to RGB for standard display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Check the actual mean of the channels; if they are close, the image is monochromatic
    print(f"Data type: {img.dtype}")
    print(f"Shape: {img.shape}")
    
    cv2.imshow("Corrected FERET", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()