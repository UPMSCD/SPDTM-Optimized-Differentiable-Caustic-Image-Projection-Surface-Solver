import os
from os.path import realpath, join

import drjit as dr
import mitsuba as mi
import time
import startupConfig
import mirrorMeshes
import optimizer
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter, median_filter


from scipy.ndimage import gaussian_filter, median_filter


mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


#initialize startup config for file location and image
config_name = 'star3'
initialConfig = startupConfig.create(config_name)
config = initialConfig.config
SCENE_DIR = initialConfig.SCENE_DIR
mi.Bitmap(config['reference'])
mi.file_resolver().append(SCENE_DIR)
##########################





#scene parameters as inputs
renderRes = (256, 256)
heihgtmapRes = (512, 512)
optimizationSteps = 300
meshSaampling = 64

mirrorTarget=[0,1,0]
mirrorOrigin=[0,0,0]

emitterTarget=[0, 0, 0]
emitterOrigin=[40, 50, 0]

imgLocation = 10


#need to use atan to get angle incident
incidentProportion =  (emitterOrigin[0] / emitterOrigin[1])
rads = math.atan(incidentProportion)
degs = math.degrees(rads)
print(f"Incident angle {degs} Degrees")
imgPosX = (imgLocation * -incidentProportion)
imgPosZ = imgLocation
imageTarget=[imgPosX, 0, 0]
imageOrigin=[imgPosX, imgPosZ, 0]

sensorPosX = imgLocation * -incidentProportion
sensorPosZ = imgLocation - 5 #at 20deg fov shift sensor back to be in perfect focus #need to figure out X placement
sensorTarget=[sensorPosX, 500, 0]
sensorOrigin=[sensorPosX, sensorPosZ, 0]
cameraFov=20






lap_loss_scale = 0
tv_loss_scale = 0.001
slope_loss_scale = 0
crash_loss_scale = 1
stray_light_loss_scale = 0
background_loss_scale = 0

clear_aperture=10.0    
clearanceAngle=7     
toolRadius=0.4  

load_obj_as_heightmap = False
load_obj_as_mesh = True
skip_upsampling = False
load_ply_as_mesh = False
align_mesh_to_ref = False
save_smoothed = False
heightmap_smooth_kick = False

###########



#
#### optimization hyper parameters
if 'PYTEST_CURRENT_TEST' not in os.environ:
    config.update({
        'render_resolution': renderRes,
        'heightmap_resolution': heihgtmapRes,
        'n_upsampling_steps': 4,
        'spp': meshSaampling,
        'max_iterations': optimizationSteps,
        'learning_rate': 3e-6,
    })




#configure output stuff 
if(load_obj_as_mesh):
    initialConfig.Outputs_obj(config) 
if(load_ply_as_mesh):
    initialConfig.Outputs_ply(config) 
if(load_obj_as_mesh == False and load_ply_as_mesh == False):
    initialConfig.Outputs(config) 
lens_fname = initialConfig.lens_fname
output_dir = initialConfig.output_dir
##############

















#################SCENE definitions
### creating emitter

emitter = None
if config['emitter'] == 'gray':
    emitter = {
        'type':'directionalarea',
        'radiance': {
            'type': 'spectrum',
            'value': 0.8
        },
    }


##creating integrator
integrator = {
    'type': 'ptracer',
    'samples_per_pass': 256,
    'max_depth': 4,
    'hide_emitters': False,
}

#assemle the scene


resx, resy = config['render_resolution']
sensor = {
    'type': 'perspective',
    'near_clip': 1,
    'far_clip': 1000,
    'fov': cameraFov,
    'to_world': \
        mi.ScalarTransform4f().look_at(
            target=sensorTarget,
            origin=sensorOrigin,
            up=[0, 0, 1]
        ),
    'sampler': {
        'type': 'independent',
        'sample_count': 512  # Not really used
    },
    'film': {
        'type': 'hdrfilm',
        'width': resx,
        'height': resy,
        'pixel_format': 'rgb',
        'rfilter': {
            # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
            'type': 'gaussian'
        }
    },
}




scene = {
    'type': 'scene',
    'sensor': sensor,
    'integrator': integrator,
    # Glass BSDF
    'simple-mirror': {
        'type': 'conductor',
    },
    'simple-glass': {
        'type': 'dielectric',
        'id': 'simple-glass-bsdf',
        'ext_ior': 'air',
        'int_ior': 1.5,
        'specular_reflectance': { 'type': 'spectrum', 'value': 0 },
    },
    'white-bsdf': {
        'type': 'diffuse',
        'id': 'white-bsdf',
        'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
    },
    'black-bsdf': {
        'type': 'diffuse',
        'id': 'black-bsdf',
        'reflectance': { 'type': 'spectrum', 'value': 0 },
    },
    # Receiving plane
    'receiving-plane': {
        'type': 'obj',
        'id': 'receiving-plane',
        'filename': 'meshes/rectangle.obj',
        'to_world': \
            mi.ScalarTransform4f().look_at(
                target=imageTarget,
                origin=imageOrigin,
                up=[0, 0, 1]
            ),
        'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
    },
    # Glass slab, excluding the 'exit' face (added separately below)

    # Glass rectangle, to be optimized
    'lens': {
        'type': 'ply',
        'id': 'lens',
        'filename': lens_fname,
        'to_world': mi.ScalarTransform4f().look_at(
            origin=mirrorOrigin,
            target=mirrorTarget,
            up=[0,0,1]
        ),
        'bsdf': {'type': 'ref', 'id': 'simple-mirror'},
    },

    # Directional area emitter placed behind the glass slab
    'focused-emitter-shape': {
        'type': 'obj',
        'filename': 'meshes/rectangle.obj',
        'to_world': mi.ScalarTransform4f().look_at(
            target=emitterTarget,
            origin=emitterOrigin,
            up=[0, 0, 1]
        ),
        'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
        'focused-emitter': emitter,
    },
}

scene = mi.load_dict(scene)

#####################################################

#print(scene)






























#visualize results
plt.ion() # Interactive mode on
fig_vis, axes_vis = plt.subplots(2, 2, figsize=(12, 10))
plt.tight_layout(pad=3.0)
fig_vis.canvas.setWindowTitle('Live Caustic Optimization & Tool Check')

def update_visualization(it, current_img, current_heightmap, reference_image, error_history, cbar,
                         aperture_mm=10.0, tool_deg=12.0, tool_rad_mm=0.4):
    """
    Updates the 2x2 grid with live render, heightmap, and red danger zones.
    """
    
    # Convert Dr.Jit to Numpy
    img_np = current_img.numpy()
    Z_scene_units = current_heightmap.numpy().squeeze()

    # 2. UNIT CONVERSION (Critical Step)
    # Your mesh is 2.0 units wide (from -1 to 1). 
    # We map that 2.0 units to 'aperture_mm'
    SCENE_WIDTH = 2.0 
    scale_factor = aperture_mm / SCENE_WIDTH

    # Convert Z to millimeters
    Z_mm = Z_scene_units * scale_factor

    # 3. CALCULATIONS (Now performed entirely in mm)
    h, w = Z_mm.shape
    pixel_pitch = aperture_mm / w  # mm per pixel
    
    # Slope (Gouging)
    dy, dx = np.gradient(Z_mm, pixel_pitch)
    slope_mag = np.sqrt(dx**2 + dy**2)
    max_slope_val = np.tan(np.radians(tool_deg))
    slope_violation = slope_mag > max_slope_val
    
    # Curvature (Fitting)
    d2y, d2x = np.gradient(dy, pixel_pitch), np.gradient(dx, pixel_pitch)
    curvature_mag = np.abs(d2y + d2x)
    max_curve_val = 1.0 / tool_rad_mm
    curve_violation = curvature_mag > max_curve_val

    # --- PLOTTING ---
    ax = axes_vis.ravel()
    
    # 1. Top Left: Current Render
    ax[0].clear()
    # Gamma correction (power 0.45) makes the faint caustics visible
    ax[0].imshow(np.clip(img_np ** 0.45, 0, 1)) 
    ax[0].set_title(f"Iter {it}: Render View")
    ax[0].axis('off')

    # 2. Top Right: Heightmap Topology (Replaced Reference)
    ax[1].clear()
    im_h = ax[1].imshow(Z_mm, cmap='viridis')
    ax[1].set_title(f"Surface Sag (mm)\nMin: {np.min(Z_mm):.3f} | Max: {np.max(Z_mm):.3f}")
    ax[1].axis('off')
    # Optional: Add colorbar only once to avoid stacking
    # if it > optimizationSteps * 0.7: 
    if(cbar):
        fig_vis.colorbar(im_h, ax=ax[1], fraction=0.046, pad=0.04)

    ax[3].clear() # Clear the previous frame's line
    ax[3].plot(error_history, color='dodgerblue', linewidth=2)
    ax[3].set_title("Total Absolute Error vs. Iteration")
    ax[3].set_xlabel("Iteration")
    ax[3].set_ylabel("Mean Absolute Error")
    ax[3].grid(True, linestyle='--', alpha=0.6)

    img_np = current_img.numpy()
    ref_np = reference_image.numpy()
    error_map = np.abs(img_np - ref_np)
    error_intensity = np.mean(error_map, axis=-1)
    #print(np.sum(error_intensity))

    # The new Error Map plot
    # Using 'inferno' or 'magma' colormap makes small errors pop
    im_err = ax[2].imshow(error_intensity, cmap='inferno')
    ax[2].set_title("Absolute Error")
    
    # Add a colorbar if it hasn't been added yet to show scale
    if not hasattr(update_visualization, 'colorbar'):
        plt.colorbar(im_err, ax=ax[2], fraction=0.046, pad=0.04)
        update_visualization.colorbar = True



    # --- CRITICAL UPDATE STEP ---
    # This forces the OS to repaint the window even if Python is busy
    fig_vis.canvas.draw()
    fig_vis.canvas.flush_events() 
    plt.pause(0.05) # Brief pause to allow the GUI to breathe


























# Make sure the reference image will have a resolution matching the sensor

sensor = scene.sensors()[0]
crop_size = sensor.film().crop_size()
image_ref = optimizer.load_ref_image(config, crop_size, output_dir=output_dir)





if(load_obj_as_heightmap):
    ###displacement texture
    initial_heightmap_resolution = config['heightmap_resolution']

    #upsampling_steps = dr.square(dr.linspace(mi.Float, 0, 1, config['n_upsampling_steps']+1, endpoint=False).numpy()[1:])
    #upsampling_steps = (config['max_iterations'] * upsampling_steps).astype(int)
    upsampling_steps = round(config['max_iterations'] / 1.2)
    print('The resolution of the heightfield will be doubled at iterations:', upsampling_steps)

    INPUT_OBJ = 'output.obj' #can change later but this output from C++ OT init
    print(f"[+] Initializing heightmap from OT result: {INPUT_OBJ}")
    ot_heights = optimizer.load_ot_as_heightmap(INPUT_OBJ, initial_heightmap_resolution)

    heightmap_texture = mi.load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'bitmap': mi.Bitmap(ot_heights),
        'raw': True,
    })

    # Actually optimized: the heightmap texture
    params = mi.traverse(heightmap_texture)
    params.keep(['data'])
    opt = mi.ad.Adam(lr=config['learning_rate'], params=params)



if(skip_upsampling):

    upsampling_steps = round(config['max_iterations'] / 1.2)
    print('The resolution of the heightfield will be doubled at iterations:', upsampling_steps)
    ###displacement texture
    initial_heightmap_resolution = config['heightmap_resolution']
    heightmap_texture = mi.load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'bitmap': mi.Bitmap(dr.zeros(mi.TensorXf, initial_heightmap_resolution)),
        'raw': True,
    })

    # Actually optimized: the heightmap texture
    params = mi.traverse(heightmap_texture)
    params.keep(['data'])
    opt = mi.ad.Adam(lr=config['learning_rate'], params=params)

if(skip_upsampling  == False and load_obj_as_heightmap == False):
    ###displacement texture
    initial_heightmap_resolution = [r // (2 ** config['n_upsampling_steps'])
                                for r in config['heightmap_resolution']]

    upsampling_steps = dr.square(dr.linspace(mi.Float, 0, 1, config['n_upsampling_steps']+1, endpoint=False).numpy()[1:])
    upsampling_steps = (config['max_iterations'] * upsampling_steps).astype(int)
    print('The resolution of the heightfield will be doubled at iterations:', upsampling_steps)

    heightmap_texture = mi.load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'bitmap': mi.Bitmap(dr.zeros(mi.TensorXf, initial_heightmap_resolution)),
        'raw': True,
    })

    # Actually optimized: the heightmap texture
    params = mi.traverse(heightmap_texture)
    params.keep(['data'])
    opt = mi.ad.Adam(lr=config['learning_rate'], params=params)




















#align mesh to the lowest loss before optimizing
def apply_alignment_and_displacement(sx, sy, sz):
    
    # Apply global translation
    # This moves the entire mesh as one solid unit
    pos_x = (initial_positions.x)
    pos_y = (initial_positions.y)
    pos_z = (initial_positions.z)

    # 2. Apply Rigid Shift (Scalar + Array)
    # Dr.Jit will automatically broadcast the scalar shift to the whole array
    new_x = initial_positions.x + sx
    new_y = initial_positions.y + sy
    new_z = initial_positions.z + sz

    # 4. Reassemble and push to Mitsuba
    # We create a new Vector3f and ravel it back to the flat [x,y,z, x,y,z...] format
    new_pos = mi.Vector3f(new_x, new_y, new_z)
    params2['lens.vertex_positions'] = dr.ravel(new_pos)
    params2.update()

if(align_mesh_to_ref):

    # 1. Setup the lens parameters
    params2 = mi.traverse(scene)
    raw_v = params2['lens.vertex_positions'] # This is the 786432 array

    num_vertices = len(raw_v) // 3
    idx = dr.arange(mi.UInt, num_vertices)

    base_x = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 0))
    base_y = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 1))
    base_z = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 2))

    initial_positions = mi.Vector3f(base_x, base_y, base_z)

    print(f"Flat size: {len(raw_v)}")
    print(f"Verified Vertex Count: {dr.shape(initial_positions.x)[0]}")


    shift_x = mi.Float(0.0)
    shift_z = mi.Float(0.0)
    shift_y = mi.Float(0.0)
    dr.enable_grad(shift_x, shift_y, shift_z)

    align_opt = mi.ad.Adam(lr=0.005)
    align_opt['sx'] = shift_x
    align_opt['sy'] = shift_y
    align_opt['sz'] = shift_z

    for i in range(25):

        apply_alignment_and_displacement(align_opt['sx'], align_opt['sy'], align_opt['sz'])
        print("[i] Phase 1: Aligning mesh position...")
        # 1. Render and Loss
        image = mi.render(scene, params2, spp=4, spp_grad=(4)) # Use slightly higher SPP for alignment
        

        image_similarity_loss = optimizer.thresholded_l2_loss(
            image=image,
            ref=image_ref
        )


        loss = image_similarity_loss

        # 2. Update ONLY the alignment optimizer
        dr.backward(loss)

        align_opt.step()
        
        print(f"  Align {i:02d} | Loss: {str(loss)} | Shift: [{align_opt['sx'][0]:.4f}, {align_opt['sy'][0]:.4f}, {align_opt['sz'][0]:.4f}]")

    # Capture the final best shift
    final_sx = dr.detach(align_opt['sx'])
    final_sz = dr.detach(align_opt['sz'])
    print(f"[+] Alignment Frozen at: X={final_sx[0]:.4f}, Z={final_sz[0]:.4f}")






















def save_smoothed_nurbs_mesh(output_path, s=0.01):
    """
    Fits a spline over the optimizer data, updates the texture, 
    applies displacement, and saves.
    """
    print(f"[i] Fitting Spline to Texture Data (s={s})...")

    # 1. Get current resolution from the optimizer tensor
    # shape is likely (H, W, 1)
    current_h, current_w, _ = dr.shape(opt['data'])
    
    # 2. Reshape and convert to NumPy for Scipy
    # We use opt['data'] as it contains the current trained heightmap
    h_map = dr.reshape(mi.TensorXf, opt['data'], (current_h, current_w))
    h_np = np.array(h_map)
    
    # 3. Fit the Bivariate Spline
    x = np.linspace(0, 1, current_h)
    z = np.linspace(0, 1, current_w)
    spline = RectBivariateSpline(x, z, h_np, kx=3, ky=3, s=s)
    
    # 4. Evaluate and convert back to Dr.Jit
    smoothed_h_np = spline(x, z).astype(np.float32)
    # Ensure it matches the (H, W, 1) shape the optimizer/texture expects
    smoothed_h_tensor = mi.TensorXf(smoothed_h_np.reshape(current_h, current_w, 1))

    # 5. TEMPORARILY overwrite the texture data
    # We detach to ensure we aren't trying to record gradients for a save operation
    opt['data'] = dr.detach(smoothed_h_tensor)
    
    # 6. Run your existing displacement logic
    # This now uses the smoothed texture data
    apply_displacement(amplitude=1.0) 
    
    # 7. Write the PLY
    lens_mesh = [m for m in scene.shapes() if m.id() == 'lens'][0]
    lens_mesh.write_ply(output_path)
    
    print(f'[+] Saved NURBS-smoothed lens to: {os.path.basename(output_path)}')



    # Calculate the difference

    mid_row = current_h // 2
    plt.plot(h_np[mid_row, :], label='Raw Optimized', alpha=0.5)
    plt.plot(smoothed_h_np[mid_row, :], label='NURBS Smoothed', linewidth=2)
    plt.legend()
    plt.title("Surface Cross-Section (Row Slice)")
    plt.ylabel("Height (mm)")
    plt.show()
    plt.pause(0.05)
    wait = input("enter?")



def interactive_surface_cleaner(opt_tensor, resolution, sigma=1.0, s_val=0):
    """
    Shows a comparison of the heightmap before and after smoothing.
    Closes the plot to accept changes and continue optimization.
    """
    # 1. Prepare Data
    h_map = dr.reshape(mi.TensorXf, opt_tensor, (resolution[0], resolution[1]))
    h_np = np.array(h_map)
    
    # 2. Apply the High-Frequency Filter (Pre-filter)
    # Median kills spikes, Gaussian smooths the rest
    h_clean = median_filter(h_np, size=3)
    h_clean = gaussian_filter(h_clean, sigma=sigma)
    
    # 3. Fit the Spline to the clean data
    x = np.linspace(0, 1, resolution[0])
    z = np.linspace(0, 1, resolution[1])
    spline = RectBivariateSpline(x, z, h_clean, kx=3, ky=3, s=s_val)
    h_smoothed = spline(x, z)

    # 4. Visualization
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot A: Raw Data
    im0 = axs[0].imshow(h_np, cmap='magma')
    axs[0].set_title("Raw (Noisy) Optimizer Data")
    fig.colorbar(im0, ax=axs[0])

    # Plot B: Smoothed Surface
    im1 = axs[1].imshow(h_smoothed, cmap='magma')
    axs[1].set_title(f"Smoothed (Sigma={sigma}, s={s_val})")
    fig.colorbar(im1, ax=axs[1])

    # Plot C: Cross-section comparison
    mid = resolution[0] // 2
    axs[2].plot(h_np[mid, :], label='Raw', alpha=0.4, color='gray')
    axs[2].plot(h_smoothed[mid, :], label='Smoothed', color='red', linewidth=2)
    axs[2].set_title("Profile Check (Middle Row)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.suptitle("Check your surface: Close window to ACCEPT and CONTINUE optimization")
    plt.show() # The script will pause here until you close the window

    # 5. Return the data to the optimizer format
    # Reshape back to (H, W, 1)
    new_data = h_smoothed.reshape(resolution[0], resolution[1], 1)
    return dr.detach(mi.TensorXf(new_data))



# --- HOW TO USE IN YOUR LOOP ---
# Every 100 iterations, or when you trigger 'kick'
#if it % 100 == 0:
#    opt['data'] = interactive_surface_cleaner(opt['data'], resolution, sigma=1.2)
#    # Re-apply to the mesh so the next render uses the smooth version
#    apply_displacement()






































































####applying displacement texture

params_scene = mi.traverse(scene)

# We will always apply displacements along the original normals and
# starting from the original positions.
positions_initial = dr.unravel(mi.Vector3f, params_scene['lens.vertex_positions'])
normals_initial   = dr.unravel(mi.Vector3f, params_scene['lens.vertex_normals'])

lens_si = dr.zeros(mi.SurfaceInteraction3f, dr.width(positions_initial))
lens_si.uv = dr.unravel(type(lens_si.uv), params_scene['lens.vertex_texcoords'])

def apply_displacement(amplitude = 1):
    # Enforce reasonable range. For reference, the receiving plane
    # is 7 scene units away from the lens.
    vmax = 100.
    params['data'] = dr.clip(params['data'], -vmax, vmax)
    dr.enable_grad(params['data'])
    params.update()

    height_values = heightmap_texture.eval_1(lens_si)
    new_positions = (height_values * normals_initial * amplitude + positions_initial)
    params_scene['lens.vertex_positions'] = dr.ravel(new_positions)
    params_scene.update()


def apply_smoothed_displacement(amplitude = 1):
    # Enforce reasonable range. For reference, the receiving plane
    # is 7 scene units away from the lens.
    vmax = 100.
    params['data'] = dr.clip(params['data'], -vmax, vmax)
    dr.enable_grad(params['data'])
    params.update()

    height_values = heightmap_texture.eval_1(lens_si)
    new_positions = (height_values * normals_initial * amplitude + positions_initial)
    params_scene['lens.vertex_positions'] = dr.ravel(new_positions)
    params_scene.update()


start_time = time.time()
mi.set_log_level(mi.LogLevel.Warn)
iterations = config['max_iterations']
loss_values = []
error_history = [0]
spp = config['spp']

itKick = 0
kicked = False
for it in range(iterations):




    t0 = time.time()

    if(heightmap_smooth_kick):
        try:
            ################################################################################################


            current_h, current_w, _ = dr.shape(opt['data'])

            h_map = dr.reshape(mi.TensorXf, opt['data'], (current_h, current_w))
            h_np = np.array(h_map)
            #h_np_filtered = gaussian_filter(h_np, sigma=1)
            h_np_filtered = h_np
            # 3. Fit the Bivariate Spline
            x = np.linspace(0, 1, current_h)
            z = np.linspace(0, 1, current_w)

        
            spline = RectBivariateSpline(x, z, h_np_filtered, kx=3, ky=3, s=0.000002, maxit=200)
            
            # 4. Evaluate and convert back to Dr.Jit
            smoothed_h_np = spline(x, z).astype(np.float32)

            smoothed_h_tensor = mi.TensorXf(smoothed_h_np.reshape(current_h, current_w, 1))

            opt['data'] = dr.detach(smoothed_h_tensor)
            

            ################################################################################################
            apply_displacement()
        except:
            print("failed to fit")
    else:
        apply_displacement()


    # Perform a differentiable rendering of the scene
    image = mi.render(scene, params, seed=it, spp=2 * spp, spp_grad=(spp * 2))

    # Scale-independent L2 function
    Z = params['data'] #heightmap tensor



    #optimization filters######
    lap_loss = optimizer.compute_laplacian(Z)
    #slope_loss = optimizer.compute_slope_penalty(Z, max_slope_tan=0.02) 
    tv_loss = optimizer.total_variation_loss(Z)
    #crash_loss = optimizer.compute_toolpath_sweep_loss_dr(Z, clear_aperture, toolRadius, clearanceAngle, toolpath)
    ##########


    total_render_energy = dr.sum(image)
    total_ref_energy = dr.sum(image_ref)
    image = image * (total_ref_energy / (total_render_energy + 1e-6)) #scale the image to use HDR range?

    # --- IMPLEMENTATION OF MULTI-SCALE LOSS ---
    # 1. Calculate the image similarity loss using the multi-scale strategy
    image_similarity_loss = optimizer.thresholded_l2_loss(
        image=image, 
        ref=image_ref
    )
    stray_light_loss = optimizer.robust_stray_light_loss(
        image=image, 
        ref=image_ref
    )

    background_mask = dr.select(image_ref < 0.05, 1.0, 0.0)
    background_loss = optimizer.background_median_loss(image, background_mask)
    bg_uniformity_loss = optimizer.background_uniformity_loss(image, background_mask)


    loss = image_similarity_loss \
        + lap_loss_scale * lap_loss \
        + tv_loss_scale * tv_loss \
        + stray_light_loss_scale * stray_light_loss \
        + background_loss_scale * background_loss \
        + background_loss_scale * bg_uniformity_loss \
        + 0 * error_history[-1] \
        #+ crash_loss_scale * crash_loss
        #+ slope_loss_scale * slope_loss
            # Back-propagate errors to input parameters and take an optimizer step
    dr.backward(loss)
    print(" " + str(loss))
    print("  bgm" + str(background_loss))    # 2. SMOOTH THE GRADIENTS (Add this)
    
    print("  bgu" + str(bg_uniformity_loss))    # 2. SMOOTH THE GRADIENTS (Add this)

    print("  tv" + str(tv_loss))    # 2. SMOOTH THE GRADIENTS (Add this)
    print("  lap" + str(lap_loss)) 
    # We treat the gradients as a texture and apply a small Gaussian blur
#    grad = dr.grad(opt['data'])
    # A 3x3 or 5x5 blur kernel usually works best
#    smoothed_grad = optimizer.gaussian_blur_2d(grad, sigma=1.5) 

    # 3. Re-assign the smoothed gradients back to the optimizer
#    dr.set_grad(opt['data'], smoothed_grad)
    # Take a gradient step


    opt.step()


    if(load_obj_as_heightmap or skip_upsampling):
        if it == upsampling_steps:
            opt['data'] = dr.upsample(opt['data'], scale_factor=(2, 2, 1))
    else:
        if it in upsampling_steps: # Increase resolution of the heightmap###########

            opt['data'] = dr.upsample(opt['data'], scale_factor=(2, 2, 1))
            #opt.set_learning_rate(2 * opt.learning_rate())
            #print(f"Increased learning rate to: {opt.learning_rate()}")


    # Carry over the update to our "latent variable" (the heightmap values)
    params.update(opt)

    # Log progress
    elapsed_ms = 1000. * (time.time() - t0)
    current_loss = loss.array[0]
    loss_values.append(current_loss)
    mi.logger().log_progress(
        it / (iterations-1),
        f'Iteration {it:03d}: loss={current_loss:g} (took {elapsed_ms:.0f}ms)',
        'Caustic Optimization', '')



    # Increase rendering quality toward the end of the optimization
    if it in (int(0.7 * iterations), int(0.9 * iterations)):
        #spp *= 2
        opt.set_learning_rate(0.5 * opt.learning_rate())

    

    #plotting error history
    img_np = dr.detach(image).numpy()
    ref_np = image_ref.numpy()
    error_map = np.abs(img_np - ref_np)
    error_intensity = np.mean(error_map, axis=-1)
    #print(np.sum(error_intensity))
    error_history.append(np.sum(error_intensity))
    if it == 4:
        error_history[0] = error_history[2]
        error_history[1] = error_history[2]



# ... inside your optimization loop ...
    if (it % 50 == 0 and it > 0):
        save = input("Save Y/N?")
        if(save == 'Y'):
                fname = join(output_dir, 'lens_displaced.ply')
                apply_displacement()
                lens_mesh = [m for m in scene.shapes() if m.id() == 'lens'][0]
                lens_mesh.write_ply(fname)
                print('[+] Saved displaced lens to:', os.path.basename(fname))

                if(save_smoothed):
                    current_h, current_w, depth = dr.shape(opt['data'])
                    
                    save_smoothed_nurbs_mesh(fname)
                    print("saveSMooth")

        if(save == "kick"):
            #opt.set_learning_rate(2 * opt.learning_rate())
            current_h, current_w, depth = dr.shape(opt['data']) # e.g., 512, 512
            low_res = (current_h // 4, current_w // 4)

            target_shape = (int(low_res[0]), int(low_res[1]), 1)
            opt['data'] = dr.resample(opt['data'], shape=target_shape)
            current_tensor = opt['data']
            new_data = dr.resample(current_tensor, shape=target_shape)
            opt['data'] = new_data
            
            itKick = it
            kicked = True
            heightmap_smooth_kick = True
        if(save == "lower"):
            opt.set_learning_rate(0.75 * opt.learning_rate())
        if(save == "TV"):
            tv_loss_scale = float(input("enter tv loss scale "))
        if(save == "NURBS"):
            heightmap_smooth_kick = True
            itKick = it
        if(save == "SMOOTH"):
            current_h, current_w, depth = dr.shape(opt['data']) # e.g., 512, 512
            check = 0.0000000000001
            beforeFilter = opt['data']
            while(check >= 0):
                opt['data'] = interactive_surface_cleaner(beforeFilter, (current_h, current_w), sigma=check)
                check = float(input("Enter nurbs Sigma: "))


    if (it - itKick > 10 and heightmap_smooth_kick):
        heightmap_smooth_kick = False

    if (it - itKick > 10 and kicked):
        #opt.set_learning_rate(0.5 * opt.learning_rate())
        opt['data'] = dr.upsample(opt['data'], scale_factor=(4, 4, 1))
        kicked = False
    #next = input("Enter to next step")



        # Visualize every 10 steps
    if it % 10 == 0 or it == (iterations - 1):
        update_visualization(
            it=it,
            current_img=dr.detach(image),           # Detach from graph
            current_heightmap=dr.detach(params['data']), # Detach from graph
            reference_image=image_ref,
            error_history=error_history,
            cbar=False,
            aperture_mm=clear_aperture,    
            tool_deg=clearanceAngle,       # Safety margin (use slightly less than 12)
            tool_rad_mm=toolRadius      
        )


save = input("Save surface Y/N?")

if(save == 'Y'):
    end_time = time.time()
    print(((end_time - start_time) * 1000) / iterations, ' ms per iteration on average')
    mi.set_log_level(mi.LogLevel.Info)

    mi.set_log_level(mi.LogLevel.Error)
    fname = join(output_dir, 'heightmap_final.exr')
    mi.util.write_bitmap(fname, params['data'])
    print('[+] Saved final heightmap state to:', os.path.basename(fname))

    fname = join(output_dir, 'lens_displaced.ply')
    apply_displacement()
    lens_mesh = [m for m in scene.shapes() if m.id() == 'lens'][0]
    lens_mesh.write_ply(fname)
    print('[+] Saved displaced lens to:', os.path.basename(fname))

    if(save_smoothed):
        save_smoothed_nurbs_mesh(fname)
        print("saveSMooth")


def load_new_mesh():
    #set up render again with new values and using exported ply as new lens mesh
    print('no')

def reoptimize():
    #run optimizer again
    print('no')




































plt.ioff() # Turn off interactive mode
plt.show() # Keep the window open until you close it


