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
optimizationSteps = 30
meshSaampling = 64

sensorTarget=[-5, 5, 0]
sensorOrigin=[-5, 2.7, 0]
cameraFov=45

imageTarget=[-5, 0, 0]
imageOrigin=[-5, 5, 0]

mirrorTarget=[0,1,0]
mirrorOrigin=[0,0,0]

emitterTarget=[0, 0, 0]
emitterOrigin=[50, 50, 0]

lap_loss_scale = 0
tv_loss_scale = 1.0005
slope_loss_scale = 0
crash_loss_scale = 1


clear_aperture=10.0    
clearanceAngle=7     
toolRadius=0.4  

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
        'learning_rate': 3e-5,
    })




#configure output stuff 
initialConfig.Outputs(config) 
lens_fname = initialConfig.lens_fname
output_dir = initialConfig.output_dir
##############




#Scaled heightmap override optimizatoin
#lens_fname = r'c:\Users\Conman569\Documents\Notes\Optics\Gregorian Design\NormalizedMetrics\Scripts\causticprojection\caustic\outputs\star\lens_Scaled_displaced.ply'
save_scaled = False

fdisplacementScale = 500
















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



























#visualize results
plt.ion() # Interactive mode on
fig_vis, axes_vis = plt.subplots(2, 2, figsize=(12, 10))
plt.tight_layout(pad=3.0)
fig_vis.canvas.setWindowTitle('Live Caustic Optimization & Tool Check')

def update_visualization(it, current_img, current_heightmap, cbar,
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




    ax[2].clear()
    ax[2].imshow(Z_mm, cmap='gray', alpha=0.4) # Background
    
    # 1. Plot Red Danger Zones
    if np.any(slope_violation):
        red_overlay = np.zeros((*slope_violation.shape, 4))
        red_overlay[slope_violation] = [1, 0, 0, 0.8] 
        ax[2].imshow(red_overlay)

    spiral_mask = optimizer.generate_spiral_overlay(h, w, revolutions=40)
    # 2. Plot Spiral Toolpath (Cyan lines)
    # Create a cyan overlay where spiral_mask is True
    spiral_overlay = np.zeros((*Z_mm.shape, 4))
    spiral_overlay[spiral_mask] = [0, 1, 1, 0.6] # Cyan, 30% opacity
    ax[2].imshow(spiral_overlay)
    
    ax[2].set_title(f"Slope Fail & Toolpath\nMax: {np.degrees(np.arctan(np.max(slope_mag))):.1f}°")
    ax[2].axis('off')

    # [Bottom Right] Radius Danger + Spiral Overlay
    ax[3].clear()
    ax[3].imshow(Z_mm, cmap='gray', alpha=0.4)
    
    if np.any(curve_violation):
        red_overlay = np.zeros((*curve_violation.shape, 4))
        red_overlay[curve_violation] = [1, 0, 0, 0.8] 
        ax[3].imshow(red_overlay)
        
    ax[3].imshow(spiral_overlay) # Add spiral here too
    ax[3].axis('off')




    # 4. Bottom Right: Radius Danger Map
    ax[3].clear()
    # Background: Faint Gray Topology
    ax[3].imshow(Z_mm, cmap='gray', alpha=0.4)
    # Overlay: Bright Red Violations
    if np.any(curve_violation):
        red_overlay = np.zeros((*curve_violation.shape, 4))
        red_overlay[curve_violation] = [1, 0, 0, 0.8] # RGBA: Red
        ax[3].imshow(red_overlay)
        
        # Calculate tightest radius in microns
        max_k = np.max(curvature_mag)
        min_r_um = (1000.0 / max_k) if max_k > 0 else 9999
        ax[3].set_title(f"RADIUS FAIL (< {tool_rad_mm*1000:.0f}µm)\nMin: {min_r_um:.0f}µm")
    else:
        ax[3].text(w/2, h/2, "PASS", color='green', fontsize=20, ha='center', weight='bold')
        ax[3].set_title(f"Radius Safety\n(Tool: {tool_rad_mm*1000:.0f}µm)")
    ax[3].axis('off')

    # --- CRITICAL UPDATE STEP ---
    # This forces the OS to repaint the window even if Python is busy
    fig_vis.canvas.draw()
    fig_vis.canvas.flush_events() 
    plt.pause(0.05) # Brief pause to allow the GUI to breathe
























# Make sure the reference image will have a resolution matching the sensor

sensor = scene.sensors()[0]
crop_size = sensor.film().crop_size()
image_ref = optimizer.load_ref_image(config, crop_size, output_dir=output_dir)




base_mesh = mirrorMeshes.create_flat_lens_mesh(heihgtmapRes)
m = mirrorMeshes.apply_optimal_transport_to_mesh(base_mesh, scene=scene, target_img=image_ref,
                                                 iterations=100, alpha=0.25,
                                                 epsilon=0.03, sinkhorn_iters=80)
m.write_ply(lens_fname)















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

def apply_displacement_scale(amplitude):
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


toolpath = optimizer.generate_spiral_toolpath_dr(clear_aperture,w=512, h=512,revolutions=12,infeed_per_rev_mm=0.02,pts_per_rev=200,start_corner='top_left')

start_time = time.time()
mi.set_log_level(mi.LogLevel.Warn)
iterations = config['max_iterations']
loss_values = []
spp = config['spp']

for it in range(iterations):



    t0 = time.time()

    # Apply displacement and update the scene BHV accordingly
    if(it == 30):
        apply_displacement_scale(50000)
    else:
        apply_displacement()


    # Perform a differentiable rendering of the scene
    image = mi.render(scene, params, seed=it, spp=2 * spp, spp_grad=spp)

    # Scale-independent L2 function
    Z = params['data'] #heightmap tensor



    #optimization filters######
    lap_loss = optimizer.compute_laplacian(Z)
    #slope_loss = optimizer.compute_slope_penalty(Z, max_slope_tan=0.02) 
    tv_loss = optimizer.total_variation_loss(Z)
    #crash_loss = optimizer.compute_toolpath_sweep_loss_dr(Z, clear_aperture, toolRadius, clearanceAngle, toolpath)
    ##########


    # --- IMPLEMENTATION OF MULTI-SCALE LOSS ---
    # 1. Calculate the image similarity loss using the multi-scale strategy
    image_similarity_loss = optimizer.thresholded_l2_loss(
        image=image, 
        ref=image_ref
    )


    loss = image_similarity_loss \
        + lap_loss_scale * lap_loss \
        + tv_loss_scale * tv_loss \
        #+ crash_loss_scale * crash_loss
        #+ slope_loss_scale * slope_loss
            # Back-propagate errors to input parameters and take an optimizer step
    dr.backward(loss)
    print(" " + str(loss))
   # print("crash: "  +str(crash_loss))


    # Take a gradient step
    opt.step()

    # Increase resolution of the heightmap
    if it in upsampling_steps:
        opt['data'] = dr.upsample(opt['data'], scale_factor=(2, 2, 1))

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

    
# ... inside your optimization loop ...

    # Visualize every 10 steps
    if it % 10 == 0:
        update_visualization(
            it=it,
            current_img=dr.detach(image),           # Detach from graph
            current_heightmap=dr.detach(params['data']), # Detach from graph
            cbar=False,
            aperture_mm=clear_aperture,    
            tool_deg=clearanceAngle,       # Safety margin (use slightly less than 12)
            tool_rad_mm=toolRadius      
        )




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




if(save_scaled):
    fname = join(output_dir, 'lens_Scaled_displaced.ply')
    apply_displacement_scale(displacementScale)
    lens_mesh = [m for m in scene.shapes() if m.id() == 'lens'][0]
    lens_mesh.write_ply(fname)
    print('[+] Saved displaced lens to:', os.path.basename(fname))


















update_visualization(
    it=it,
    current_img=dr.detach(image),           # Detach from graph
    current_heightmap=dr.detach(params['data']), # Detach from graph
    cbar=True,
    aperture_mm=clear_aperture,    
    tool_deg=clearanceAngle,       # Safety margin (use slightly less than 12)
    tool_rad_mm=toolRadius         
)











plt.ioff() # Turn off interactive mode
plt.show() # Keep the window open until you close it


