
import drjit as dr
import mitsuba as mi

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


def ot_initialize_mesh(mesh, scene, target_img, alpha=0.25):
    sensor = scene.sensors()[0]

    # 1. Face centroids in screen
    face_xy = compute_face_centroids_screen(mesh, sensor, scene)

    # 2. Current flux per face
    face_flux = compute_face_flux(scene, mesh)
    face_flux /= dr.sum(face_flux)

    # 3. Target distribution
    target_flux = target_flux_from_image(target_img)

    # Pixel coordinates
    res_x, res_y = target_flux.shape
    px, py = dr.meshgrid(
        dr.linspace(mi.Float, 0, res_x, res_x),
        dr.linspace(mi.Float, 0, res_y, res_y),
        indexing='ij'
    )
    pixel_xy = dr.stack([dr.ravel(px), dr.ravel(py)], axis=1)
    pixel_flux = dr.ravel(target_flux)

    # 4. Sinkhorn OT
    Pmat = sinkhorn_ot(
        face_xy, face_flux,
        pixel_xy, pixel_flux,
        epsilon=0.03,
        iters=80
    )

    # 5. OT centroids
    face_ot_targets = compute_ot_barycenters(Pmat, pixel_xy, face_flux)

    # 6. Pull mesh
    new_mesh = update_mesh_vertices(mesh, face_ot_targets, sensor, alpha)

    return new_mesh



def update_mesh_vertices(mesh, face_ot_targets, scene_sensor, alpha=0.25):
    params = mi.traverse(mesh)
    verts = dr.unravel(mi.Vector3f, params["vertex_positions"])
    faces = dr.unravel(mi.Vector3u, params["faces"])

    # Pull each triangle’s vertices towards the inverse-projection of its OT target
    face3d = scene_sensor.unproject(face_ot_targets)  # If sensor has no unproject, I can provide an impl.

    v0 = dr.gather(mi.UInt, faces.x, dr.arange(mi.UInt, faces.x.size()))
    v1 = dr.gather(mi.UInt, faces.y, dr.arange(mi.UInt, faces.x.size()))
    v2 = dr.gather(mi.UInt, faces.z, dr.arange(mi.UInt, faces.x.size()))

    for fi in range(faces.x.size()):
        target = face3d[fi]

        verts[v0[fi]] += alpha * (target - verts[v0[fi]])
        verts[v1[fi]] += alpha * (target - verts[v1[fi]])
        verts[v2[fi]] += alpha * (target - verts[v2[fi]])

    params["vertex_positions"] = dr.ravel(verts)
    params.update()
    return mesh

def compute_ot_barycenters(Pmat, pixel_xy, face_flux):
    """
    Returns (F,2) OT centroids for each face.
    """
    # Weighted sum over pixel_xy
    cx = dr.dot(Pmat, pixel_xy[:, 0])
    cy = dr.dot(Pmat, pixel_xy[:, 1])

    return dr.stack([cx / face_flux, cy / face_flux], axis=1)

def sinkhorn_ot(face_xy, face_flux, pixel_xy, pixel_flux, epsilon=0.01, iters=120):
    """
    face_xy: (F,2)
    face_flux: (F,)
    pixel_xy: (P,2)
    pixel_flux: (P,)
    returns transport matrix P_{ij} shape (F,P)
    """
    F = face_xy.shape[0]
    Pn = pixel_xy.shape[0]

    # Cost matrix C_ij = ||x_i - y_j||^2
    x = face_xy[:, None, :]       # (F,1,2)
    y = pixel_xy[None, :, :]      # (1,P,2)
    C = dr.squared_norm(x - y)    # (F,P)

    # Gibbs kernel
    K = dr.exp(-C / epsilon)

    u = dr.full(mi.Float, 1.0 / F, F)
    v = dr.full(mi.Float, 1.0 / Pn, Pn)

    # Sinkhorn iterations
    for _ in range(iters):
        u = face_flux / dr.dot(K, v)
        v = pixel_flux / dr.dot(K.T, u)

    # Final transport matrix
    Pmat = u[:, None] * K * v[None, :]
    return Pmat


def target_flux_from_image(target_img):
    # convert to luminance
    T = mi.math.mean(target_img, axis=2)
    T = dr.maximum(T, 1e-8)
    T_sum = dr.sum(T)
    return T / T_sum




def compute_face_flux(scene, mesh):
    """
    Renders per-face contributions using Mitsuba AD.
    Assumes a white target and direct illumination.
    Returns flux per face: shape (n_faces,)
    """
    integrator = mi.load_dict({"type": "path"})
    img = integrator.render(scene)

    # Convert image to scalar luminance
    lum = mi.math.mean(img, axis=2)

    res_x, res_y = lum.shape

    # Rasterize triangle areas into the luminance texels.
    # Simple version: sample luminance at centroid pixel:
    params = mi.traverse(mesh)
    faces = dr.unravel(mi.Vector3u, params["faces"])
    centroids = compute_face_centroids_screen(mesh, scene.sensors()[0], scene)

    cx = dr.clamp(dr.floor(centroids[:, 0]), 0, res_x - 1)
    cy = dr.clamp(dr.floor(centroids[:, 1]), 0, res_y - 1)

    lum_1D = dr.ravel(lum)
    idx = cy * res_x + cx
    face_flux = dr.gather(mi.Float, lum_1D, idx)

    return face_flux


def compute_face_centroids_screen(mesh, sensor, scene):
    """
    Returns (n_faces,2) pixel coordinates of screen-space centroids.
    Uses manual projection (compatible with Mitsuba-3).
    """

    params = mi.traverse(mesh)
    verts = dr.unravel(mi.Vector3f, params['vertex_positions'])
    faces = dr.unravel(mi.Vector3u, params['faces'])

    v0 = dr.gather(mi.Vector3f, verts, faces.x)
    v1 = dr.gather(mi.Vector3f, verts, faces.y)
    v2 = dr.gather(mi.Vector3f, verts, faces.z)

    c3d = (v0 + v1 + v2) / 3.0

    xy = project_world_to_pixel(c3d, sensor)

    return xy



# --- helper: unproject pixels to 3D points on a plane z = plane_z (world coords) ---
def unproject_pixels_to_plane(sensor, pixel_xy, plane_z=0.0):
    """
    pixel_xy: (N,2) dr or numpy array of pixel coords (x_px, y_px) in pixel space
    Returns: (N,3) dr.Vector3f array of points on the plane z=plane_z in world coords.

    This constructs camera rays through pixels and intersects them with the Z=plane_z plane
    in world coordinates. Works for perspective sensors.
    """
    # Ensure pixel_xy is numpy for indexing convenience
    import numpy as _np

    # Film resolution
    res_x, res_y = sensor.film().crop_size()

    # Convert pixel indices -> normalized device coords in [0,1]
    # Accept both dr arrays and numpy arrays
    if isinstance(pixel_xy, dr.TensorXf) or isinstance(pixel_xy, dr.Array):
        px = dr.unravel(mi.Float, pixel_xy[:,0])
        py = dr.unravel(mi.Float, pixel_xy[:,1])
        # convert to numpy for math below
        px_np = px.numpy()
        py_np = py.numpy()
    else:
        px_np = _np.asarray(pixel_xy[:,0])
        py_np = _np.asarray(pixel_xy[:,1])

    # NDC in [0,1]
    ndc_x = (px_np + 0.5) / float(res_x)
    ndc_y = (py_np + 0.5) / float(res_y)

    # Convert NDC -> screen coords used by Mitsuba sensor (usually -1..1)
    # For perspective sensor: x_screen = (2*ndc_x - 1) * aspect * tan(fov/2)
    # We'll construct rays in camera space then transform by sensor->world
    import math
    cam_to_world = sensor.world_transform() if hasattr(sensor, 'world_transform') else sensor.to_world

    # Get fov in radians (assume 'fov' parameter exists)
    try:
        fov = float(sensor.fov)
    except Exception:
        # fallback: assume 45 deg
        fov = 45.0
    fov_rad = math.radians(fov)
    # aspect ratio
    aspect = float(res_x) / float(res_y)

    # screen coordinates in camera space (x_c, y_c) at z = -1 (camera looks towards -z in some conventions)
    # We choose camera forward -Z and near plane at z = -1 unit in camera space for ray direction construction.
    # Map ndc to screen plane: x_s = (2*ndc_x - 1) * tan(fov/2) * aspect, y_s = (1 - 2*ndc_y) * tan(fov/2)
    tan_half = math.tan(fov_rad * 0.5)
    x_s = (2.0 * ndc_x - 1.0) * tan_half * aspect
    y_s = (1.0 - 2.0 * ndc_y) * tan_half

    # camera ray origins and directions in camera space:
    # origin at (0,0,0), direction = normalize([x_s, y_s, -1.0])
    dirs = _np.stack([x_s, y_s, -_np.ones_like(x_s)], axis=1)
    lens = _np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (lens + 1e-12)

    # Transform origin and direction to world space using sensor->world transform matrix
    # Mitsuba exposes to_world as a ScalarTransform4f; retrieve matrix
    # We'll try to use sensor.to_world.matrix() if available, otherwise use provided transform
    try:
        M = sensor.to_world.matrix()  # 4x4
        M = _np.array(M)  # convert to numpy
    except Exception:
        # try different attribute
        try:
            M = sensor.world_transform().matrix()
            M = _np.array(M)
        except Exception:
            raise RuntimeError("Cannot obtain sensor to_world transform for unprojection.")

    # Camera origin in world coords
    origin_world = _np.dot(M, _np.array([0.0, 0.0, 0.0, 1.0]))[:3]

    # Transform directions: rotate by upper-left 3x3 of M (no translation)
    R = M[:3, :3]
    dirs_world = dirs @ R.T  # (N,3)

    # Ray-plane intersection: plane z = plane_z. Ray: p(t) = origin_world + t * dir_world
    # Solve for t where p_z(t) = plane_z
    oz = origin_world[2]
    dz = dirs_world[:, 2]
    # prevent divide by zero
    t = (plane_z - oz) / (dz + 1e-12)

    pts = origin_world[None, :] + dirs_world * t[:, None]  # (N,3)

    # Convert to dr.Vector3f
    pts_dr = dr.unravel(mi.Vector3f, dr.asarray(pts.astype(_np.float32)).ravel())

    # Return as (N,3) dr.Tensor
    # Reshape to (N,3) container: we return a list-like dr array shaped (N, 3)
    N = pts.shape[0]
    pts3 = dr.reshape(pts_dr, (N, 3))
    return pts3



# --- apply_optimal_transport_to_mesh wrapper ---
def apply_optimal_transport_to_mesh(mesh, scene=None, target_img=None, iterations=80, alpha=0.25,
                                    epsilon=0.03, sinkhorn_iters=80):
    """
    Wrapper that runs OT initialization and returns an updated mesh.
    - mesh: mitsuba Mesh (flat initial)
    - scene: mitsuba Scene object (required for compute_face_flux and projection)
    - target_img: mitsuba Bitmap/texture or numpy image (if None, tries to load scene reference or returns mesh unchanged)
    - iterations: (unused in this simple wrapper) kept for API compatibility
    - alpha: step size when pulling vertices
    """
    # require scene and target image for meaningful OT initialization
    if scene is None or target_img is None:
        print("[mirrorMeshes] apply_optimal_transport_to_mesh: scene or target_img is None; skipping OT.")
        return mesh

    # If target_img is a path string, try to load with Mitsuba
    if isinstance(target_img, str):
        target_img = mi.Bitmap(target_img)

    # If target_img is a mitsuba Bitmap, convert to dr tensor
    # If it's a numpy array already, keep it
    # Call ot_initialize_mesh (expects mesh, scene, target_img)
    new_mesh = ot_initialize_mesh(mesh, scene, target_img, alpha=alpha)
    return new_mesh


# --- small edit: update_mesh_vertices to use unprojection helper ---
def update_mesh_vertices(mesh, face_ot_targets, scene_sensor, alpha=0.25):
    params = mi.traverse(mesh)
    verts = dr.unravel(mi.Vector3f, params["vertex_positions"])
    faces = dr.unravel(mi.Vector3u, params["faces"])

    # Pull each triangle’s vertices towards the inverse-projection of its OT target
    # face_ot_targets is in pixel coordinates. Map them to 3D on plane z=0 (mesh base plane)
    face3d = unproject_pixels_to_plane(scene_sensor, face_ot_targets, plane_z=0.0)

    # gather vertex indices
    n_faces = faces.x.size()
    # iterate per-face (explicit python loop) since mesh sizes are moderate for initialization
    for fi in range(n_faces):
        target = face3d[fi]  # dr.Vector3f
        v0_idx = int(faces.x[fi].numpy())
        v1_idx = int(faces.y[fi].numpy())
        v2_idx = int(faces.z[fi].numpy())

        verts[v0_idx] = verts[v0_idx] + alpha * (target - verts[v0_idx])
        verts[v1_idx] = verts[v1_idx] + alpha * (target - verts[v1_idx])
        verts[v2_idx] = verts[v2_idx] + alpha * (target - verts[v2_idx])

    params["vertex_positions"] = dr.ravel(verts)
    params.update()
    return mesh



def project_world_to_pixel(points_world, sensor):
    """
    Manual Mitsuba-3 projection on CPU/GPU/AD. 
    points_world: (N, 3) Vector3f
    returns (N, 2) pixel coords
    """

    # Camera transforms
    cam_to_world = sensor.world_transform()
    world_to_cam = dr.inverse(cam_to_world)

    # Film resolution
    res_x, res_y = sensor.film().crop_size()

    # FOV (horizontal)
    fov = sensor.parameters()['fov']
    tan_half_fov = dr.tan(dr.radians(fov) * 0.5)

    # Promote to homogeneous coordinates
    pw = dr.concat(points_world, dr.full(mi.Float, 1.0, points_world.x.shape))

    # Transform world → camera space
    pc = world_to_cam @ pw                 # shape (N, 4)
    xc = pc.x
    yc = pc.y
    zc = pc.z  # positive if in front of camera

    # Perspective divide to get camera NDC
    ndc_x =  (xc / (zc * tan_half_fov))
    ndc_y = -(yc / (zc * tan_half_fov)) * (res_x / res_y)   # keep aspect ratio

    # Convert NDC [-1,1] → pixel coordinates
    px = (ndc_x * 0.5 + 0.5) * res_x
    py = (ndc_y * 0.5 + 0.5) * res_y

    return dr.stack([px, py], axis=1)
