import mitsuba as mi
import numpy as np
import scipy.ndimage as nd
import cv2
import numpy.fft as fft
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')

# -----------------------------
# Load the original heightmap
# -----------------------------
img = mi.Bitmap("heightmap_final.exr")
Z = np.array(img, dtype=np.float64).squeeze()

# -----------------------------
# Convert to millimeters
# -----------------------------
aperture_mm = 10.0  # clear aperture size in mm
N = Z.shape[0]

# 1 scene unit = aperture_mm / N mm
scene_to_mm = aperture_mm / N * 50000
Z = Z * scene_to_mm

# -----------------------------
# Filtering pipeline (now in mm)
# -----------------------------
# Gaussian smoothing
Z_smooth = nd.gaussian_filter(Z, sigma=3)

# Bilateral filtering
Z_bilat = cv2.bilateralFilter(
    Z_smooth.astype(np.float32),
    d=9, sigmaColor=0.01, sigmaSpace=3
).astype(np.float64)

# FFT low-pass filtering
def lowpass_filter(Z, cutoff_ratio=0.2):
    F = fft.fftshift(fft.fft2(Z))
    nx, ny = Z.shape
    x = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    mask = R < cutoff_ratio
    F_filtered = F * mask
    return np.real(fft.ifft2(fft.ifftshift(F_filtered)))

Z_filtered = lowpass_filter(Z_bilat, cutoff_ratio=0.1)

# Clip final sag (optional, in mm)
Z_final = np.clip(Z_filtered, -50.05, 50.05)  # ±50 µm

# -----------------------------
# Coordinates in mm for XYZ export
# -----------------------------
dx = aperture_mm / (N - 1)
x = (np.arange(N) - (N - 1)/2) * dx
y = (np.arange(N) - (N - 1)/2) * dx
X, Y = np.meshgrid(x, y)

pts = np.column_stack((X.flatten(), Y.flatten(), Z_final.flatten()))
np.savetxt("heightmap_mm.xyz", pts, fmt="%.6f")

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(Z, cmap='viridis')
plt.colorbar(label="Height (mm)")
plt.title("Original heightmap (mm)")

plt.subplot(2, 2, 2)
plt.imshow(Z_smooth, cmap='viridis')
plt.colorbar(label="Height (mm)")
plt.title("Gaussian filtered (mm)")

plt.subplot(2, 2, 3)
plt.imshow(Z_bilat, cmap='viridis')
plt.colorbar(label="Height (mm)")
plt.title("Bilateral filtered (mm)")

plt.subplot(2, 2, 4)
plt.imshow(Z_filtered, cmap='viridis')
plt.colorbar(label="Height (mm)")
plt.title("FFT low-pass filtered (mm)")

plt.figure(figsize=(5,4))
plt.imshow(Z_final, cmap='viridis')
plt.colorbar(label="Height (mm)")
plt.title("Final clipped sag (±50 µm)")
plt.tight_layout()
plt.show()
