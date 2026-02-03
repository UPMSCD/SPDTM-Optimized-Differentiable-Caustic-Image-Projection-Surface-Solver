import numpy as np
from plyfile import PlyData

def read_scaled_ply_to_xyz(filename, target_mm=10.0):
    """Reads PLY, scales to target aperture, and returns XYZ."""
    plydata = PlyData.read(filename)
    v = plydata['vertex'].data
    
    # 1. Extract raw coordinates
    # We use your mapping: PLY(x, z, y) -> XYZ(X, Y, Z)
    raw_x = v['x']
    raw_z = v['z']
    raw_y = v['y']
    
    # 2. Calculate Current Aperture (Width)
    # This finds the span of the lens in scene units
    current_width = np.max(raw_x) - np.min(raw_x)
    
    # 3. Calculate Scale Factor
    scale_factor = target_mm / current_width
    
    print(f"[i] Detected Width: {current_width:.6f} units")
    print(f"[i] Scaling by: {scale_factor:.6f}x to reach {target_mm}mm")
    
    # 4. Apply Scaling and Center at (0,0)
    # Centering makes it much easier to setup in CAM software
    new_x = (raw_x - np.mean(raw_x)) * scale_factor
    new_y = (raw_z - np.mean(raw_z)) * scale_factor
    new_z = (raw_y - np.mean(raw_y)) * scale_factor # Height/Depth
    
    return np.column_stack([new_x, new_y, new_z])

# --- Run the conversion ---
pts = read_scaled_ply_to_xyz('lens_displaced.ply', target_mm=10.0)

# Save the final result
output_name = "ConvertedPLY_10mm.xyz"
np.savetxt(output_name, pts, fmt="%.6f")

print(f"[+] Saved to {output_name}")
print(f"[i] Final Z-depth (Max Sag): {np.max(pts[:,2]) - np.min(pts[:,2]):.4f} mm")