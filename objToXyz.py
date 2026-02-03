import numpy as np

def extract_mirror_xyz(input_file, output_file, target_aperture_mm=10.0, decimal_precision=3):
    vertices = []

    # 1. Read the OBJ file
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        print("No vertices found.")
        return

    points = np.array(vertices)

    # 2. Filter for the "Top" surface only
    # We group points by (x, y) and keep the one with the highest Z.
    # We round x and y slightly to account for floating point errors in the OBJ.
    surface_map = {}
    for p in points:
        # Create a key based on x and y rounded to a high precision
        key = (round(p[0], decimal_precision), round(p[1], decimal_precision))
        
        if key not in surface_map or p[2] > surface_map[key][2]:
            surface_map[key] = p

    mirror_points = np.array(list(surface_map.values()))

    # 3. Center the mirror (Optional but recommended for optics)
    # This puts the center of the mirror at (0,0)
    avg_x = (np.min(mirror_points[:, 0]) + np.max(mirror_points[:, 0])) / 2
    avg_y = (np.min(mirror_points[:, 1]) + np.max(mirror_points[:, 1])) / 2
    mirror_points[:, 0] -= avg_x
    mirror_points[:, 1] -= avg_y

    # 4. Scale to the defined aperture
    current_width = np.max(mirror_points[:, 0]) - np.min(mirror_points[:, 0])
    scale_factor = target_aperture_mm / current_width
    scaled_points = mirror_points * scale_factor

    # 5. Save to XYZ
    np.savetxt(output_file, scaled_points, fmt='%.6f', delimiter=' ')

    print(f"Extraction Complete!")
    print(f"Original points: {len(points)} | Mirror surface points: {len(scaled_points)}")
    print(f"Final Aperture: {target_aperture_mm}mm")

# --- Settings ---
INPUT_OBJ = "output.obj" 
OUTPUT_XYZ = "convertedOBJ.xyz"
TARGET_WIDTH = 10.0 # mm

if __name__ == "__main__":
    extract_mirror_xyz(INPUT_OBJ, OUTPUT_XYZ, TARGET_WIDTH)