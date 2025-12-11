import drjit as dr
import mitsuba as mi
import math
from os.path import realpath, join
import numpy as np

def compute_laplacian(Z):
    """Discrete Laplacian for smoothness penalty, compatible with older Dr.Jit."""
    # Shift up
    Z_up = dr.concat([Z[-1:, :], Z[:-1, :]], axis=0)
    # Shift down
    Z_down = dr.concat([Z[1:, :], Z[:1, :]], axis=0)
    # Shift left
    Z_left = dr.concat([Z[:, -1:], Z[:, :-1]], axis=1)
    # Shift right
    Z_right = dr.concat([Z[:, 1:], Z[:, :1]], axis=1)

    lap = Z_up + Z_down + Z_left + Z_right - 4 * Z

    return dr.mean(dr.square(lap)) 


def compute_slope_penalty(Z, max_slope_tan):
    """
    Penalizes slopes that exceed the tool clearance angle.
    max_slope_tan: tan(12 degrees) ~= 0.212
    """
    # 1. Calculate gradient (local slope)
    dz_dy = Z[1:, :] - Z[:-1, :]
    dz_dx = Z[:, 1:] - Z[:, :-1]

    # 2. Calculate magnitude of the slope
    # We use vector length: sqrt(dx^2 + dy^2)
    # Note: Pad to maintain shape consistency
    dz_dy = dr.concat((dz_dy, dz_dy[-1:, :]), axis=0)
    dz_dx = dr.concat((dz_dx, dz_dx[:, -1:]), axis=1)
    
    slope_mag = dr.sqrt(dr.square(dz_dx) + dr.square(dz_dy))

    # 3. Penalize only values EXCEEDING the limit
    # This acts like a ReLU: 0 if safe, increases linearly if unsafe
    excess = dr.maximum(slope_mag - max_slope_tan, 0.0)
    
    # Square the excess to make the penalty aggressive as it gets worse
    return dr.mean(dr.square(excess))



def total_variation_loss(Z, epsilon=1e-8):
    diff_x = Z[1:, :, :] - Z[:-1, :, :]   # shape (H-1, W, 1)
    diff_y = Z[:, 1:, :] - Z[:, :-1, :]   # shape (H, W-1, 1)

    # Compute TV as sum of magnitudes of diffs separately, then sum
    tv_x = dr.sum(dr.sqrt(diff_x**2 + epsilon))
    tv_y = dr.sum(dr.sqrt(diff_y**2 + epsilon))

    return tv_x + tv_y


#### load reference image 
def load_ref_image(config, resolution, output_dir):
    b = mi.Bitmap(config['reference'])
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, False)
    if dr.any(b.size() != resolution):
        b = b.resample(resolution)

    mi.util.write_bitmap(join(output_dir, 'out_ref.exr'), b)

    print('[i] Loaded reference image from:', config['reference'])
    return mi.TensorXf(b)


def scale_independent_loss(image, ref):
    """Brightness-independent L2 loss function."""
    scaled_image = image / dr.mean(dr.detach(image))
    scaled_ref = ref / dr.mean(ref)
    return dr.mean(dr.square(scaled_image - scaled_ref))








def compute_curvature_penalty(Z, pixel_size_mm, tool_radius_mm): ###should use this to fit tool radius in surface
    """
    Penalizes surface curvature sharper than the tool radius.
    curvature (k) ~ 1/R. 
    If k > 1/R_tool, the tool cannot fit.
    """
    # Max allowed curvature (kappa)
    k_max = 1.0 / tool_radius_mm

    # Discrete Laplacian (approximation of curvature)
    # Note: You already have a compute_laplacian function, utilize it.
    # However, raw Laplacian is unitless regarding pixel scale.
    # We must scale by pixel size squared (dx^2) to get real units (1/mm).
    
    lap = compute_laplacian(Z) 
    
    # Scale Laplacian to physical curvature [1/mm]
    # curvature ~ laplacian / (pixel_width_mm)^2
    # You need to pass the physical size of one pixel here.
    k_surface = dr.abs(lap) / (pixel_size_mm ** 2)

    # Penalize curvature that is tighter than the tool
    excess = dr.maximum(k_surface - k_max, 0.0)
    
    return dr.mean(dr.square(excess))


def compute_fabrication_loss(Z, aperture_mm, resolution_x, tool_radius_mm, clearance_deg):
    """
    Computes a loss that penalizes surface features impossible to diamond turn.
    """
    # 1. Calculate Pixel Pitch (Distance between pixels in mm)
    # We detach this because we don't want to differentiate the resolution
    pixel_pitch_mm = aperture_mm / float(resolution_x)
    
    # ---------------------------------------------------------
    # CONSTRAINT A: SLOPE (Prevents Conical Flank Gouging)
    # ---------------------------------------------------------
    # Calculate local slope (rise over run)
    # We divide by pixel_pitch_mm to convert "Height per Pixel" to "Height per mm"
    dz_dx = (Z[:, 1:] - Z[:, :-1]) / pixel_pitch_mm
    dz_dy = (Z[1:, :] - Z[:-1, :]) / pixel_pitch_mm
    
    # Pad to maintain shape
    dz_dx = dr.concat((dz_dx, dz_dx[:, -1:]), axis=1)
    dz_dy = dr.concat((dz_dy, dz_dy[-1:, :]), axis=0)
    
    # Gradient Magnitude (The steepness at any point)
    grad_mag = dr.sqrt(dr.square(dz_dx) + dr.square(dz_dy))
    
    # Max allowed slope = tan(clearance_angle)
    # For 12 degrees, this is approx 0.212
    max_slope = dr.tan(dr.deg2rad(clearance_deg))
    
    # ReLU Penalty: Only punish if slope > max_slope
    slope_excess = dr.maximum(grad_mag - max_slope, 0.0)
    slope_loss = dr.mean(dr.square(slope_excess))

    # ---------------------------------------------------------
    # CONSTRAINT B: CURVATURE (Prevents Tool Radius Bridging)
    # ---------------------------------------------------------
    # We use the Laplacian as an approximation of Mean Curvature.
    # To be physically correct, Curvature = Laplacian / pixel_pitch^2
    
    # Shifted arrays for Laplacian calculation
    Z_up    = dr.concat([Z[-1:, :], Z[:-1, :]], axis=0)
    Z_down  = dr.concat([Z[1:, :], Z[:1, :]], axis=0)
    Z_left  = dr.concat([Z[:, -1:], Z[:, :-1]], axis=1)
    Z_right = dr.concat([Z[:, 1:], Z[:, :1]], axis=1)
    
    # Discrete Laplacian
    laplacian = (Z_up + Z_down + Z_left + Z_right - 4.0 * Z)
    
    # Convert to Physical Curvature (units: 1/mm)
    # k = d2z/dx2
    curvature = laplacian / (pixel_pitch_mm ** 2)
    
    # Max allowed curvature = 1 / Tool Radius
    # e.g., for 0.4mm tool, max curvature is 2.5 (1/mm)
    max_curvature = 1.0 / tool_radius_mm
    
    # IMPORTANT: We only care about CONCAVE valleys.
    # Depending on your Z-axis direction, concave might be positive or negative.
    # Assuming standard conventions, we usually penalize the absolute magnitude 
    # to be safe, or just the positive side if Z points up.
    # Using absolute value is the safest conservative approach.
    curvature_excess = dr.maximum(dr.abs(curvature) - max_curvature, 0.0)
    curvature_loss = dr.mean(dr.square(curvature_excess))

    return slope_loss, curvature_loss


def log_l2_loss(image, ref, epsilon=1e-5):
    # Compress dynamic range before comparing
    log_img = dr.log(image + epsilon)
    log_ref = dr.log(ref + epsilon)
    
    # Normalize brightness in log space (optional but good for caustics)
    log_img = log_img - dr.mean(log_img)
    log_ref = log_ref - dr.mean(log_ref)
    
    return dr.mean(dr.square(log_img - log_ref))

# You need a gaussian blur function. 
# Since Dr.Jit doesn't have a built-in conv2d easy access, 
# we can simulate it by downsampling (averaging) which is cheaper/faster.

def multi_scale_loss(image, ref, iteration, max_iterations): #############no workie downsampling
    """
    Early iterations: Compare heavily downsampled (blurry) images.
    Late iterations: Compare full resolution images.
    """
    # Calculate progress 0.0 to 1.0
    prog = iteration / max_iterations
    
    loss = 0.0
    
    # ALWAYS add the base resolution loss (so fine details matter)
    loss += scale_independent_loss(image, ref)
    
    # IF we are in the first 50% of training, add a coarse guide
    if prog < 0.5:
        # Downsample by 4x (averaging 4x4 blocks) effectively blurs the image
        # This allows the optimizer to match "large blobs" of light
        img_small = dr.upsample(image, scale_factor=(4, 4, 1))
        ref_small = dr.upsample(ref,   scale_factor=(4, 4, 1))
        
        # Add this coarse loss with a weight
        loss += 0.5 * scale_independent_loss(img_small, ref_small)
        
    return loss


def thresholded_l2_loss(image, ref, epsilon=0.01):
    """Calculates L2 loss, ignoring differences smaller than epsilon near the target."""
    
    # Calculate the raw difference
    diff = image - ref
    
    # Create masks to identify pixels near pure black (ref=0) or pure white (ref=1)
    is_black = dr.abs(ref) < 1e-6
    is_white = dr.abs(ref - 1.0) < 1e-6
    
    # Zero out the gradient if the pixel is slightly noisy but close to the target:
    
    # 1. If target is BLACK, but rendered pixel is near-black (0 < image < epsilon)
    #    -> We want to ignore this slight noise.
    condition_near_black = is_black & (image < epsilon)
    
    # 2. If target is WHITE, but rendered pixel is near-white (1-epsilon < image < 1)
    #    -> We want to ignore this slight noise.
    condition_near_white = is_white & (image > (1.0 - epsilon))
    
    # If either condition is met, set the difference (and thus the squared loss) to zero.
    diff = dr.select(condition_near_black | condition_near_white, 0.0, diff)
    
    return dr.mean(dr.sqr(diff))




def generate_spiral_overlay(res_x, res_y, revolutions=50):
    """
    Generates a boolean mask of a spiral toolpath.
    """
    import numpy as np
    y, x = np.ogrid[:res_y, :res_x]
    
    # Normalize to -1 to 1
    x = (x / res_x) * 2 - 1
    y = (y / res_y) * 2 - 1
    
    # Polar coordinates
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Spiral Equation: r = a + b * theta
    # We want 'revolutions' turns.
    # Normalize theta to be continuous increasing
    theta_continuous = theta + np.pi # 0 to 2pi
    
    # Calculate phase of spiral
    # We want r to match the phase of the spiral
    # The toolpath exists where (r * constant) % 1 is close to 0
    
    # Scaling factor to make 'revolutions' rings
    spiral_phase = (r * revolutions) - (theta / (2*np.pi))
    
    # Create lines where the phase is close to an integer
    thickness = 0.1 # Adjust for line thickness
    mask = np.abs(spiral_phase - np.round(spiral_phase)) < thickness
    
    return mask



















# -------------------------
# Spiral generator (Archimedean)
# -------------------------
def generate_spiral_toolpath_dr(aperture_mm,
                                w=512, h=512,
                                revolutions=12,
                                infeed_per_rev_mm=0.02,
                                pts_per_rev=200,
                                start_corner='top_left'):
    """
    Returns:
      tool_xy_mm: (K,2) numpy array of tool center positions in mm (x,y) relative to scene center (0,0)
      tangents: (K,2) numpy array of tangent unit vectors at each tool center
      pixel_coords: (K,2) numpy array of pixel coordinates (x_px, y_px) for convenience
    Notes:
      - The coordinate frame is centered: x,y in [-aperture_mm/2, aperture_mm/2].
      - The spiral runs from the (approx) corner distance inward to center.
      - The sampling density is revolutions * pts_per_rev points.
    """
    # total number of points
    K = int(revolutions * pts_per_rev)
    theta = np.linspace(0.0, 2.0 * math.pi * revolutions, K)

    # Archimedean infeed per radian
    f = infeed_per_rev_mm / (2.0 * math.pi)  # mm per rad

    # Maximum radius: use corner distance (center -> corner)
    corner_dist = math.sqrt(2.0) * (aperture_mm / 2.0)
    r_max = corner_dist

    # Build decreasing radius (edge -> center): r = r_max - f * theta
    r = r_max - f * theta
    # clamp to >= 0
    r = np.maximum(r, 0.0)

    # apply corner rotation offset to start near requested corner
    offsets = {'top_left': 3*np.pi/4, 'top_right': np.pi/4,
               'bottom_right': -np.pi/4, 'bottom_left': -3*np.pi/4}
    ang_off = offsets.get(start_corner, 0.0)
    theta_shifted = theta + ang_off

    x_mm = r * np.cos(theta_shifted)
    y_mm = r * np.sin(theta_shifted)

    # Tangents: derivative wrt theta -> dx/dtheta, dy/dtheta
    # dx/dtheta = dr/dtheta * cos - r * sin ; dr/dtheta = -f
    dr_dtheta = -f
    dx_dtheta = dr_dtheta * np.cos(theta_shifted) - r * np.sin(theta_shifted)
    dy_dtheta = dr_dtheta * np.sin(theta_shifted) + r * np.cos(theta_shifted)

    # unit tangent vectors
    t_norm = np.sqrt(dx_dtheta**2 + dy_dtheta**2) + 1e-12
    tx = dx_dtheta / t_norm
    ty = dy_dtheta / t_norm

    # Convert to pixel coords (use same convention as your visualization code)
    pixel_pitch = aperture_mm / float(w)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    px = x_mm / pixel_pitch + cx
    py = y_mm / pixel_pitch + cy

    tool_xy_mm = np.stack([x_mm, y_mm], axis=1)   # (K,2)
    tangents = np.stack([tx, ty], axis=1)
    pixel_coords = np.stack([px, py], axis=1)

    return tool_xy_mm, tangents, pixel_coords


# -------------------------
# Full tool geometry profile builder (sphere + cone)
# -------------------------
def build_tool_profile_R_alpha(R_mm, alpha_deg):
    alpha = math.radians(alpha_deg)
    r_t = R_mm * math.cos(alpha)             # tangency radius
    a = R_mm * math.cos(2.0*alpha) / math.sin(alpha)
    # Returns a numpy/dr-friendly function that maps radial distances (mm) -> vertical offset (mm)
    def S_tool(r_mm):
        # r_mm may be numpy or dr array; use numpy for precompute in Python, dr in main compute if desired
        # Use broadcasting safe operations:
        return dr.where(r_mm <= r_t,
                        -dr.sqrt(dr.clip(R_mm*R_mm - r_mm*r_mm, 0.0, None)),
                        -a + r_mm / dr.tan(alpha))
    return S_tool, r_t, a


# -------------------------
# Main Dr.Jit swept-volume loss (batched centers)
# -------------------------
def compute_toolpath_sweep_loss_dr(Z_scene_units,
                                   aperture_mm,
                                   tool_radius_mm,
                                   clearance_angle_deg,
                                   toolpath_xy_mm,    # numpy (K,2) in mm
                                   batch_centers=64,
                                   softmin_tau=0.5,
                                   kernel_extra_R=3.0,
                                   pad_mode='edge',
                                   penalty_scale=1.0):
    """
    Dr.Jit-compatible implementation. Returns dr scalar loss and diagnostics.
    Z_scene_units: dr.TensorXf or numpy array of shape (H,W) in scene units (same as params['data'])
    aperture_mm: total physical width (mm) mapping 2 scene units -> aperture_mm
    toolpath_xy_mm: numpy array of (K,2) centers in mm relative to center
    batch_centers: how many centers to process per batch (trade memory/time)
    kernel_extra_R: multiplies R to define search kernel radius e.g. R*(1+kernel_extra_R)
    Returns:
      total_loss (dr.scalar), dict with swept_surface (dr array HxW), violation mask
    """
    # Convert input to dr arrays if needed
    # If Z_scene_units is a numpy array, convert to dr tensor. If it's already dr, leave it.
    if isinstance(Z_scene_units, np.ndarray):
        Z = dr.asarray(Z_scene_units)
    else:
        Z = Z_scene_units

    H_px = Z.shape[0]
    W_px = Z.shape[1]
    pixel_pitch = aperture_mm / float(W_px)   # mm / pixel

    # Map scene units -> mm (your previous scaling: scene_width=2 -> aperture_mm)
    # I.e., scene unit * (aperture_mm/2.0) -> mm
    Z_mm = Z * (aperture_mm / 2.0)

    # Build tool profile
    S_tool_func, r_t, a = build_tool_profile_R_alpha(tool_radius_mm, clearance_angle_deg)

    # Kernel radius to sample offsets (in mm)
    kernel_radius_mm = tool_radius_mm + kernel_extra_R * tool_radius_mm
    radius_px = int(math.ceil(kernel_radius_mm / pixel_pitch))

    # Precompute offset grid (integer pixel offsets)
    xs = np.arange(-radius_px, radius_px + 1, dtype=np.int32)
    ys = np.arange(-radius_px, radius_px + 1, dtype=np.int32)
    DX, DY = np.meshgrid(xs, ys, indexing='xy')  # arrays in numpy
    dist_mm = np.sqrt((DX * pixel_pitch)**2 + (DY * pixel_pitch)**2)
    inside_mask = dist_mm <= kernel_radius_mm
    off_dx = DX[inside_mask].astype(np.int32)   # shape (n_offsets,)
    off_dy = DY[inside_mask].astype(np.int32)
    off_dist_mm = dist_mm[inside_mask].astype(np.float32)
    n_offsets = off_dx.shape[0]

    # Precompute profile values (numpy)
    # we use dr for actual autograd path but these constant offsets/profile_vals are fine as numpy constants
    # For dr compatibility we will convert profile_vals to dr arrays when needed
    profile_vals_np = np.empty(n_offsets, dtype=np.float32)
    # compute piecewise
    R = tool_radius_mm
    alpha = math.radians(clearance_angle_deg)
    r_t_np = R * math.cos(alpha)
    a_np = R * math.cos(2.0*alpha) / math.sin(alpha)
    for i in range(n_offsets):
        rr = off_dist_mm[i]
        if rr <= r_t_np:
            profile_vals_np[i] = -math.sqrt(max(R*R - rr*rr, 0.0))
        else:
            profile_vals_np[i] = -a_np + rr / math.tan(alpha)
    profile_vals = dr.asarray(profile_vals_np)  # shape (n_offsets,)

    # Pad Z_mm (dr pad may not be available; do pad with numpy then convert)
    # We'll convert Z_mm to numpy for padding, then back to dr. This keeps the padded values constants with respect to grads.
    # NOTE: if Z_mm is a dr array that contains grads, converting to numpy will detach; so instead implement padding using dr.concat if you want grads on borders.
    # Here we assume boundary padding 'edge' is acceptable to implement with dr.concatenate (so grads flow inside).
    # Implement pad in dr:
    pad = radius_px
    # build padded array by concatenation along axes using dr operations:
    # Left/Right pads
    left_cols = dr.repeat(dr.unsqueeze(Z_mm[:, 0], 1), pad, axis=1)   # H x pad
    right_cols = dr.repeat(dr.unsqueeze(Z_mm[:, -1], 1), pad, axis=1)
    Z_lr_padded = dr.concat([left_cols, Z_mm, right_cols], axis=1)   # H x (W + 2pad)
    # Now pad top/bottom
    top_rows = dr.repeat(dr.unsqueeze(Z_lr_padded[0, :], 0), pad, axis=0)    # pad x (W+2pad)
    bottom_rows = dr.repeat(dr.unsqueeze(Z_lr_padded[-1, :], 0), pad, axis=0)
    Zpad = dr.concat([top_rows, Z_lr_padded, bottom_rows], axis=0)  # (H+2pad) x (W+2pad)

    # Helper to extract shifted image slice using numpy integer slices but gathering from dr array.
    # We'll compute slices in terms of start indices (row_start, col_start) into Zpad.
    Hpad = H_px + 2*pad
    Wpad = W_px + 2*pad

    # To build shifted images for a given center offset in numpy loops we'll use dr.gather from flattened indices.
    # Build flattened coordinates grid of target image (H x W) in padded coords origin (0..Hpad-1, 0..Wpad-1)
    yy = np.arange(H_px) + pad  # rows in padded coord corresponding to original image rows
    xx = np.arange(W_px) + pad
    YY, XX = np.meshgrid(yy, xx, indexing='ij')  # shape (H,W)
    base_indices = (YY * Wpad + XX).astype(np.int32)  # flattened indices shape (H,W)
    base_indices_flat = base_indices.ravel()  # (H*W,)

    # Convert Zpad to flattened dr vector for dr.gather
    Zpad_flat = dr.ravel(Zpad)  # length Hpad*Wpad

    # Convert offsets to numpy arrays (already off_dx, off_dy)
    off_dx_np = off_dx
    off_dy_np = off_dy

    # Prepare storage for swept contribution across centers in batches
    K = int(toolpath_xy_mm.shape[0])
    # We'll compute the swept surface by computing, for a batch of centers:
    #  - for each offset i: sample shifted image (via base_indices + (dy_i * Wpad + dx_i))
    #  - candidate center-height images = shifted - profile_vals[i]
    #  - soft-min over offsets -> zcmin_k (H,W)
    #  - build A_k = zcmin_k + S_tool(dist_grid_from_center)
    # then we take soft-min across centers in the batch and combine with previous batches.

    # precompute pixel centers in mm
    xs_mm = dr.linspace(mi.Float, -aperture_mm/2.0, aperture_mm/2.0, W_px) if hasattr(dr, 'linspace') else dr.asarray(np.linspace(-aperture_mm/2.0, aperture_mm/2.0, W_px, dtype=np.float32))
    ys_mm = dr.linspace(mi.Float, -aperture_mm/2.0, aperture_mm/2.0, H_px) if hasattr(dr, 'linspace') else dr.asarray(np.linspace(-aperture_mm/2.0, aperture_mm/2.0, H_px, dtype=np.float32))
    Xg = dr.reshape(dr.repeat(xs_mm[None, :], H_px, axis=0), (H_px, W_px))
    Yg = dr.reshape(dr.repeat(ys_mm[:, None], W_px, axis=1), (H_px, W_px))

    # Initialize swept surface accumulator as a large positive field (we will take min)
    INF = 1e6
    swept_accum = dr.full(mi.Float, INF, (H_px, W_px))

    # convert base_indices_flat to dr int array for gather
    base_idx_dr = dr.asarray(base_indices_flat.astype(np.int32))

    # Precompute flat offset increments for gather: offset_flat_delta = dy_i * Wpad + dx_i
    offset_flat_deltas = off_dy_np * Wpad + off_dx_np     # numpy array length n_offsets
    offset_flat_deltas = offset_flat_deltas.astype(np.int32)
    # Convert to dr
    offset_flat_deltas_dr = dr.asarray(offset_flat_deltas)

    # Batch over centers
    K_total = K
    k = 0
    tau = float(softmin_tau)
    while k < K_total:
        k_end = min(K_total, k + batch_centers)
        batch = toolpath_xy_mm[k:k_end]   # numpy slice
        batch_size = batch.shape[0]

        # For each center in batch, compute integer pixel coordinates in padded grid
        cx = (W_px - 1) / 2.0
        cy = (H_px - 1) / 2.0
        # toolpath coordinates are (x_mm, y_mm) relative to center
        tx_mm = batch[:, 0]
        ty_mm = batch[:, 1]
        # Convert tools mm -> pixel coords
        tx_px = (tx_mm / pixel_pitch) + cx
        ty_px = (ty_mm / pixel_pitch) + cy
        # round to nearest pixel center (we assume pixel-snap centers)
        tx_px_i = np.round(tx_px).astype(np.int32)
        ty_px_i = np.round(ty_px).astype(np.int32)

        # For each center we need to compute stacked candidate center heights for each offset.
        # We'll produce a (batch_size, n_offsets, H*W) array of candidates (flattened), but to save memory we'll generate per-offset and accumulate a soft-min numerically stable.
        # Initialize zcmin_flat_batch as +inf
        zcmin_flat_batch = dr.full(mi.Float, INF, (batch_size, H_px * W_px))

        # We'll compute for each offset i: gather values = Zpad_flat[ base_idx + delta_i + center_shift ]
        # center_shift accounts for the tool center pixel being not at the image center: shift = (ty_px_i - cy)*Wpad + (tx_px_i - cx)
        # But since base_idx is already computed for the canonical (no shift) case, shift per center = (ty_px_i - cy) * Wpad + (tx_px_i - cx)  -> integer

        # Compute center shifts for gather (numpy)
        shift_per_center = (ty_px_i - int(cy)) * Wpad + (tx_px_i - int(cx))   # shape (batch_size,)
        # Convert to dr
        shift_per_center_dr = dr.asarray(shift_per_center.astype(np.int32))  # (batch_size,)

        # Now loop over offsets (this is n_offsets loops; n_offsets typically ~ few thousand but this loop only does gather + arithmetic)
        # If n_offsets too large consider reducing kernel_extra_R or doing vectorized offset batching
        for i_off in range(n_offsets):
            delta = int(offset_flat_deltas[i_off])   # python int
            # candidate_flat for each center: for center c: gather indices = base_idx_dr + delta + shift_per_center[c]
            # We can compute gather indices for batch by broadcasting: base_idx_flat + delta + shift_c
            # base_idx_dr is shape (H*W,)
            # We need indices shape (batch_size, H*W) -> we compute per-center by adding shift scalar then gather.
            # Create per-center indices via dr.repeat of base_idx and adding scalar shift
            base_rep = dr.repeat(base_idx_dr[None, :], batch_size, axis=0)  # (batch_size, H*W)
            shift_vec = dr.expand(shift_per_center_dr, (1, base_rep.shape[1]))  # broadcast
            gather_idx = base_rep + int(delta) + shift_vec  # (batch_size, H*W)
            # clamp gather_idx (safety)
            gather_idx_clamped = dr.clamp(gather_idx, 0, int(Hpad * Wpad - 1))
            # gather values
            vals = dr.gather(mi.Float, Zpad_flat, gather_idx_clamped)  # (batch_size, H*W)
            # subtract profile (constant per offset)
            vals_minus_profile = vals - float(profile_vals_np[i_off])
            # update soft-min: use log-sum-exp style accumulation
            # we compute zcmin = -tau * log(sum(exp(-vals/tau)))
            # but we want to accumulate across offsets; easier: store exponentials sum
            # To avoid large memory we do incremental log-sum-exp style: maintain m = min, s = sum(exp(-(x-m)/tau))
            if i_off == 0:
                # initialize m and s
                m = vals_minus_profile
                s = dr.exp(-(vals_minus_profile - m) / tau)  # = 1
                # store m,s as arrays
            else:
                # perform stable log-sum-exp merge of current (m,s) with new values "v"
                v = vals_minus_profile
                # new_m = minimum(m, v)
                new_m = dr.minimum(m, v)
                # new_s = s * exp(-(m - new_m)/tau) + exp(-(v - new_m)/tau)
                new_s = s * dr.exp(-(m - new_m) / tau) + dr.exp(-(v - new_m) / tau)
                m = new_m
                s = new_s
            # After finishing all offsets, zcmin_flat_batch = -tau * log(s) + m

        # final zcmin_flat_batch for this batch
        zcmin_flat_batch = -tau * dr.log(s + 1e-12) + m  # shape (batch_size, H*W)

        # reshape and compute allowed A_k for each center in batch:
        zcmin_batch = dr.reshape(zcmin_flat_batch, (batch_size, H_px, W_px))

        # For each center c compute dist grid_mm from that center to each pixel
        # We'll build dist per-center using pixel grid and center mm coords
        # center mm coords:
        tx_mm_batch = dr.asarray(tx_mm.astype(np.float32))
        ty_mm_batch = dr.asarray(ty_mm.astype(np.float32))
        # Expand to (batch_size, H, W)
        Xg_rep = dr.repeat(dr.unsqueeze(Xg, 0), batch_size, axis=0)   # (batch_size, H, W)
        Yg_rep = dr.repeat(dr.unsqueeze(Yg, 0), batch_size, axis=0)
        cx_batch = tx_mm_batch[:, None, None]
        cy_batch = ty_mm_batch[:, None, None]
        dx_grid = Xg_rep - cx_batch
        dy_grid = Yg_rep - cy_batch
        dist_grid_mm = dr.sqrt(dx_grid*dx_grid + dy_grid*dy_grid)

        # compute S_tool(dist) using S_tool_func but S_tool_func is dr-aware
        Sdist = S_tool_func(dist_grid_mm)   # (batch_size, H, W)

        A_batch = zcmin_batch + Sdist   # (batch_size, H, W)

        # Now reduce across centers in this batch: soft-min across batch_size
        # We'll do the same log-sum-exp soft-min across centers for stability
        # Convert to shape (batch_size, H*W) for same approach
        A_flat = dr.reshape(A_batch, (batch_size, H_px * W_px))
        # Compute soft-min across rows:
        # Use stable trick: mC = min(A_flat across axis 0)
        mC = dr.min(A_flat, axis=0)   # shape (H*W,)
        sC = dr.sum(dr.exp(-(A_flat - mC[None,:]) / tau), axis=0)
        batch_swept_flat = -tau * dr.log(sC + 1e-12) + mC
        batch_swept = dr.reshape(batch_swept_flat, (H_px, W_px))

        # Merge with global swept_accum via soft-min (we keep it as hard min for simplicity)
        swept_accum = dr.minimum(swept_accum, batch_swept)

        # advance
        k = k_end

    # Final swept_accum is the soft-approx swept surface (mm)
    swept_surface_mm = swept_accum

    # Compute violation: Z_mm - swept (positive means material above allowed)
    violation = dr.clip(Z_mm - swept_surface_mm, 0.0, None)
    gouge_loss = dr.mean(dr.sqr(violation))

    total_loss = penalty_scale * gouge_loss
    return total_loss, dict(swept_surface_mm=swept_surface_mm, violation=violation)
