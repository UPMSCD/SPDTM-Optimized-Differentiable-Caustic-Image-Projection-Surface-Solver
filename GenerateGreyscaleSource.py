import numpy as np
import cv2

def generate_source_gradient(filename="source_gradient.png", res=1024):
    # 1. Create a linear range from 1.0 (left) to 0.7 (right)
    # This represents the falloff in light intensity across the mirror
    line = np.linspace(1.0, 0.7, res)
    
    # 2. Project this 1D line into a 2D square grid
    gradient_2d = np.tile(line, (res, 1))
    
    # 3. Convert to 8-bit image data (0-255)
    # 1.0 -> 255
    # 0.7 -> 178
    img_data = (gradient_2d * 255).astype(np.uint8)
    
    # 4. Save as PNG
    cv2.imwrite(filename, img_data)
    print(f"[+] Successfully generated: {filename}")

if __name__ == "__main__":
    generate_source_gradient()