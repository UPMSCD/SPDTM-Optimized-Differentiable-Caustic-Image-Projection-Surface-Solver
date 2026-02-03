import mitsuba as mi
import numpy as np
import scipy.ndimage as nd


mi.set_variant('cuda_ad_rgb')

img = mi.Bitmap("heightmap_final.exr")
Z = np.array(img, dtype=np.float64).squeeze()

Z_smooth = nd.gaussian_filter(Z, sigma=3)

import cv2
Z_bilat = cv2.bilateralFilter(Z_smooth.astype(np.float32), 
                              d=9, sigmaColor=0.01, sigmaSpace=3)
Z_bilat = Z_bilat.astype(np.float64)


import numpy.fft as fft

def lowpass_filter(Z, cutoff_ratio=0.2):
    F = fft.fftshift(fft.fft2(Z))
    nx, ny = Z.shape
    x = np.linspace(-1,1,nx)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    mask = R < cutoff_ratio
    F_filtered = F * mask

    return np.real(fft.ifft2(fft.ifftshift(F_filtered)))

Z_filtered = lowpass_filter(Z_bilat, cutoff_ratio=0.1)


Z_final = np.clip(Z_filtered, -0.05, 0.05)



import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

# 1 — original heightmap
plt.subplot(2, 2, 1)
plt.imshow(Z, cmap='viridis')
plt.colorbar()
plt.title("Original heightmap")

# 2 — Gaussian filtered
plt.subplot(2, 2, 2)
plt.imshow(Z_smooth, cmap='viridis')
plt.colorbar()
plt.title("Gaussian filtered (sigma=3)")

# 3 — Bilateral filtered
plt.subplot(2, 2, 3)
plt.imshow(Z_bilat, cmap='viridis')
plt.colorbar()
plt.title("Bilateral filtered")

# 4 — FFT low-pass filtered (tool-limited)
plt.subplot(2, 2, 4)
plt.imshow(Z_filtered, cmap='viridis')
plt.colorbar()
plt.title("FFT low-pass filtered (cutoff=0.1)")

plt.figure(figsize=(5,4))
plt.imshow(Z_final, cmap='viridis')
plt.colorbar()
plt.title("Final clipped sag (±50 µm)")

plt.tight_layout()
plt.show()

