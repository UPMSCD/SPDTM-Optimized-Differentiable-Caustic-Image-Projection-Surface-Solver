
import drjit as dr
import mitsuba as mi
import numpy as np
import trimesh



def create_flat_lens_mesh(resolution):
    # Generate UV coordinates
    U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, resolution[0]),
        dr.linspace(mi.Float, 0, 1, resolution[1]),
        indexing='ij'
    )
    texcoords = mi.Vector2f(U, V)

    # Generate vertex coordinates
    X = 2.0 * (U - 0.5)
    Y = 2.0 * (V - 0.5)
    vertices = mi.Vector3f(X, Y, 0.0)

    # Create two triangles per grid cell
    faces_x, faces_y, faces_z = [], [], []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1
            faces_x.extend([v00, v01])
            faces_y.extend([v10, v10])
            faces_z.extend([v01, v11])

    # Assemble face buffer
    faces = mi.Vector3u(faces_x, faces_y, faces_z)

    # Instantiate the mesh object
    mesh = mi.Mesh("lens-mesh", resolution[0] * resolution[1], len(faces_x), has_vertex_texcoords=True)

    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['vertex_texcoords'] = dr.ravel(texcoords)
    mesh_params['faces'] = dr.ravel(faces)
    mesh_params.update()

    return mesh



def create_mesh_from_ot_obj(input_obj, resolution=(512, 512), target_width=2.0):
    """
    Extracts the top surface of a 3D OBJ and creates a clean 2D PLY mesh
    mapped to Mitsuba's coordinate system.
    """
    import trimesh # 'pip install trimesh' is recommended for robust mesh slicing
    
    # 1. Load the full OBJ
    mesh = trimesh.load(input_obj)
    
    # 2. Identify 'Top' faces (those with normals pointing mostly in +Z)
    # This filters out the sides and bottom of the cube
    top_face_indices = np.where(mesh.face_normals[:, 2] > 0.5)[0]
    top_surface = mesh.submesh([top_face_indices], append=True)
    
    # 3. Clean up the geometry
    # Center it and scale to Mitsuba units (usually -1 to 1)
    vertices = top_surface.vertices
    avg_xy = (np.min(vertices[:, :2], axis=0) + np.max(vertices[:, :2], axis=0)) / 2
    vertices[:, :2] -= avg_xy
    
    current_width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    scale = target_width / current_width
    vertices *= scale
    
    # 4. Convert to Mitsuba Mesh
    mi_mesh = mi.Mesh(
        "ot-top-surface",
        vertex_count=len(vertices),
        face_count=len(top_surface.faces),
        has_vertex_texcoords=True
    )
    
    # Create UVs based on the XY extents
    u = (vertices[:, 0] - np.min(vertices[:, 0])) / (np.max(vertices[:, 0]) - np.min(vertices[:, 0]))
    v = (vertices[:, 1] - np.min(vertices[:, 1])) / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    texcoords = np.stack([u, v], axis=-1)
    
    v_np = np.array(vertices.T, dtype=np.float32, order='C')
    uv_np = np.array(texcoords.T, dtype=np.float32, order='C')
    f_np = np.array(top_surface.faces.T, dtype=np.uint32, order='C')

    # 2. Update Mitsuba Mesh buffers
    params = mi.traverse(mi_mesh)
    
    # We use dr.ravel on the initialized Mitsuba Vectors
    params['vertex_positions'] = dr.ravel(mi.Vector3f(v_np))
    params['vertex_texcoords'] = dr.ravel(mi.Vector2f(uv_np))
    params['faces'] = dr.ravel(mi.Vector3u(f_np))
    
    params.update()
    
    return mi_mesh



def create_mesh_from_ply(input_ply, target_width=2.0):
    """
    Loads a PLY file (likely from a previous optimization checkpoint),
    extracts the top surface, and prepares it for Mitsuba.
    """
    # 1. Load the PLY mesh
    # trimesh handles .ply (binary or ascii) automatically
    mesh = trimesh.load(input_ply)
    
    # 2. Identify 'Top' faces
    # If your PLY is already just a 2D sheet, this will select all faces.
    # If it's a solid, it peels the top.
    face_normals = mesh.face_normals


    
    top_surface = mesh



    avg_normal_z = np.mean(top_surface.face_normals[:, 2])
    if avg_normal_z > 0:
        print("[!] Normals appear flipped. Correcting...")
        # Flip the winding order of the faces to flip the normals
        top_surface.faces = np.fliplr(top_surface.faces)
    
    # 3. Clean up and Center
    vertices = top_surface.vertices
    vertices[:, 0] *= -1 #flipping left/right because the z was backwards and inverted

    
    # 2. Scale
    current_width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    scale = target_width / current_width
    vertices *= scale

    # 3. CRITICAL: Swap Y and Z to move from XY-plane to XZ-plane
    # We want (X, Y, Z) -> (X, Z, Y)
    # This makes the "sag" happen on the Y-axis to match your working bbox [min_y=0]
    aligned_vertices = np.zeros_like(vertices)
    aligned_vertices[:, 0] = vertices[:, 0] # X stays X
    aligned_vertices[:, 1] = vertices[:, 2] # Z-sag moves to Y-axis
    aligned_vertices[:, 2] = vertices[:, 1] # Y moves to Z-axis

    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)
    v_range = v_max - v_min

    u = (vertices[:, 0] - v_min[0]) / v_range[0]
    v = (vertices[:, 2] - v_min[2]) / v_range[2]

    texcoords = np.stack([u, v], axis=-1)

    

    # 5. Build the Mitsuba Mesh
    mi_mesh = mi.Mesh(
        "ply-top-surface",
        vertex_count=len(vertices),
        face_count=len(top_surface.faces),
        has_vertex_texcoords=True
    )

    # 6. Memory Layout Correction (The C-order Fix)
    v_np = np.array(aligned_vertices.T, dtype=np.float32, order='C')
    uv_np = np.array(texcoords.T, dtype=np.float32, order='C')
    f_np = np.array(top_surface.faces.T, dtype=np.uint32, order='C')



    params = mi.traverse(mi_mesh)
    params['vertex_positions'] = dr.ravel(mi.Vector3f(v_np))
    params['vertex_texcoords'] = dr.ravel(mi.Vector2f(uv_np))
    params['faces'] = dr.ravel(mi.Vector3u(f_np))
    params.update()
    
    return mi_mesh


























