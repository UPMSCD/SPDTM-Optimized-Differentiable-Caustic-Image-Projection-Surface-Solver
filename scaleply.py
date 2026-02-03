import numpy as np
import scipy.ndimage as nd
import cv2
import numpy.fft as fft
import os
from plyfile import PlyData, PlyElement # Requires: pip install plyfile

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\Conman569\Documents\Notes\Optics\Gregorian Design\NormalizedMetrics\Scripts\causticprojection\caustic\outputs\star\lens_displaced.ply"
OUTPUT_FILE = r"C:\Users\Conman569\Documents\Notes\Optics\Gregorian Design\NormalizedMetrics\Scripts\causticprojection\caustic\outputs\star\output.ply"
SCALE_FACTOR = 50000.0
# --- FILTER PARAMETERS (Matches your previous script) ---
GAUSSIAN_SIGMA = 3
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 0.01
BILATERAL_SIGMA_SPACE = 3
FFT_CUTOFF_RATIO = 0.1
# ---------------------

def lowpass_filter(Z_grid, cutoff_ratio):
    """Applies a 2D FFT low-pass filter to the heightmap grid."""
    F = fft.fftshift(fft.fft2(Z_grid))
    nx, ny = Z_grid.shape
    x = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    mask = R < cutoff_ratio
    F_filtered = F * mask
    return np.real(fft.ifft2(fft.ifftshift(F_filtered)))

def process_ply_mesh():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at '{INPUT_FILE}'")
        print("Please check the file name and path.")
        return

    print(f"Loading mesh from: {INPUT_FILE}...")
    try:
        plydata = PlyData.read(INPUT_FILE)
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return

    # Extract vertex data element (usually named 'vertex')
    vertices = plydata['vertex'].data

    # Extract X, Y, Z coordinates
    X_flat = vertices['x']
    Y_flat = vertices['y']
    Z_original_flat = vertices['z']
    N_vertices = len(vertices)

    # 1. Infer the square grid size N
    N = int(np.sqrt(N_vertices))
    if N * N != N_vertices:
        print(f"Warning: Number of vertices ({N_vertices}) is not a perfect square. Cannot reshape for 2D filtering.")
        return

    print(f"Mesh loaded: {N_vertices} vertices. Assuming N={N} grid for filtering.")

    # 2. Rescale Z (The height deviation)
    # Applying the scale factor of 50000x directly to the Z coordinates
    Z_scaled_flat = Z_original_flat * SCALE_FACTOR

    # 3. Reshape Z into an N x N grid for 2D filtering
    Z_grid = Z_scaled_flat.reshape(N, N)
    
    # 4. Apply Filtering Pipeline (All in the new scaled units)
    print("Applying Gaussian smoothing...")
    Z_smooth = nd.gaussian_filter(Z_grid, sigma=GAUSSIAN_SIGMA)

    print("Applying Bilateral filtering...")
    # cv2.bilateralFilter requires float32 input
    Z_bilat = cv2.bilateralFilter(
        Z_smooth.astype(np.float32),
        d=BILATERAL_D, 
        sigmaColor=BILATERAL_SIGMA_COLOR, 
        sigmaSpace=BILATERAL_SIGMA_SPACE
    ).astype(np.float64) # Convert back to float64

    print("Applying FFT Low-Pass filtering...")
    Z_filtered_grid = lowpass_filter(Z_bilat, cutoff_ratio=FFT_CUTOFF_RATIO)

    # 5. Flatten the filtered Z back to a 1D array
    Z_final_flat = Z_filtered_grid.flatten()

    # 6. Create the new vertex array structure
    # The new vertex array must maintain the same 'dtype' as the original
    new_vertices = np.empty(N_vertices, vertices.dtype)
    new_vertices['x'] = X_flat
    new_vertices['y'] = Y_flat
    new_vertices['z'] = Z_final_flat # The only modified column

    # 7. Create the new PlyElement for vertices
    el_vertex = PlyElement.make(new_vertices, name='vertex')

    # 8. Rebuild the PlyData object, maintaining faces and other elements
    elements = [el_vertex]
    for el in plydata.elements:
        if el.name != 'vertex':
            elements.append(el)
            
    new_plydata = PlyData(elements, text=plydata.text)

    # 9. Write the final mesh
    new_plydata.write(OUTPUT_FILE)
    print(f"\nSuccess! Filtered and scaled mesh saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_ply_mesh()