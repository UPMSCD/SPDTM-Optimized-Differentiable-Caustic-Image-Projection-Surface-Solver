import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

# --- CONFIGURATION ---
FILE_PATH = r"C:\Users\Conman569\Documents\Notes\Optics\Gregorian Design\NormalizedMetrics\Scripts\causticprojection\caustic\outputs\star\heightmap_final.exr" # Or load your processed numpy array
APERTURE_MM = 10.0                # Total width of the optic
TOOL_CLEARANCE_DEG = 12.0
TOOL_RADIUS_MM = 0.400
# ---------------------

# Load Data
bmp = mi.Bitmap(FILE_PATH)
Z = np.array(bmp).squeeze() 

# Convert Scene Units to Millimeters (Use the logic from your export script)
N = Z.shape[0]
pixel_pitch_mm = APERTURE_MM / N
# Assuming Z in the EXR is already scaled to scene units, convert to mm:
# (Adjust this conversion factor to match your export script exactly)
scene_to_mm = (APERTURE_MM / N) #* 50000 
Z_mm = Z * scene_to_mm

# 1. CALCULATE SLOPE (Gradient)
gy, gx = np.gradient(Z_mm, pixel_pitch_mm)
slope_mag = np.sqrt(gx**2 + gy**2)
max_allowed_slope = np.tan(np.radians(TOOL_CLEARANCE_DEG))

# 2. CALCULATE CURVATURE (Laplacian)
# Curvature k = approx Laplacian
del2 = np.gradient(np.gradient(Z_mm, pixel_pitch_mm)[0], pixel_pitch_mm)[0] + \
       np.gradient(np.gradient(Z_mm, pixel_pitch_mm)[1], pixel_pitch_mm)[1]
       
# Radius of curvature R = 1/k. 
# We care about concave valleys (sign depends on your Z direction).
# Generally, we just look at absolute curvature magnitude for the "fit" check.
curvature_mag = np.abs(del2)
max_allowed_curvature = 1.0 / TOOL_RADIUS_MM

# 3. VISUALIZE ERRORS
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot Height
ax[0].imshow(Z_mm)
ax[0].set_title("Height Map (mm)")

# Plot Slope Violations
slope_mask = slope_mag > max_allowed_slope
ax[1].imshow(slope_mask, cmap='Reds', interpolation='nearest')
ax[1].set_title(f"Clearance Violations (> {TOOL_CLEARANCE_DEG}°)\nRed = Tool Flank Hit")

# Plot Radius Violations
curv_mask = curvature_mag > max_allowed_curvature
ax[2].imshow(curv_mask, cmap='Reds', interpolation='nearest')
ax[2].set_title(f"Radius Violations (< {TOOL_RADIUS_MM*1000:.0f}µm)\nRed = Tool Won't Fit")

plt.show()

if np.any(slope_mask):
    print(f"FAIL: Max slope found was {np.degrees(np.arctan(np.max(slope_mag))):.2f} degrees.")
else:
    print("PASS: Slope is within clearance angle.")

if np.any(curv_mask):
    print(f"FAIL: Tightest curvature radius was {1000/np.max(curvature_mag):.2f} microns (Tool: {TOOL_RADIUS_MM*1000} microns).")
else:
    print("PASS: Curvature is smoother than tool radius.")