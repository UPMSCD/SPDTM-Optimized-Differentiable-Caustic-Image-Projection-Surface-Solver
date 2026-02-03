import mitsuba as mi
import numpy as np


mi.set_variant('cuda_ad_rgb')
img = mi.Bitmap("heightmap_final.exr")
height = np.array(img, dtype=np.float64).squeeze()

aperture_mm = 10.0   # your clear aperture size in millimeters

N = height.shape[0]

dx = aperture_mm / (N - 1)   # mm per pixel


x = (np.arange(N) - (N - 1)/2) * dx
y = (np.arange(N) - (N - 1)/2) * dx
X, Y = np.meshgrid(x, y)

Z = height   # already in same units as Mitsuba transform scale


pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
np.savetxt("heightmap.xyz", pts, fmt="%.6f")


