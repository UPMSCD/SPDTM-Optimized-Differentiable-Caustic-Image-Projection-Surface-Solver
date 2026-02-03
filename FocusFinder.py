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

import FocusTranslator
from scipy.optimize import shgo, dual_annealing
from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
optimizationSteps = 100
meshSaampling = 64



mirrorTarget=[0,1,0]
mirrorOrigin=[0,0,0]

emitterTarget=[0, 0, 0]
emitterOrigin=[0.1, 50, 0]


import math
#need to use atan to get angle incident
incidentProportion =  (emitterOrigin[0] / emitterOrigin[1])
rads = math.atan(incidentProportion)
degs = math.degrees(rads)
print(f"Incident angle {degs} Degrees")

imgLocation = 0.1

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
tv_loss_scale = 1.0005
slope_loss_scale = 0
crash_loss_scale = 1


clear_aperture=10.0    
clearanceAngle=7     
toolRadius=0.4  

load_obj_as_heightmap = False
load_obj_as_mesh = True
load_ply_as_mesh = False

align_mesh_to_ref = True


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
if(load_obj_as_mesh):
    initialConfig.Outputs_obj(config) 
else:
    initialConfig.Outputs(config) 
if(load_ply_as_mesh):
    initialConfig.Outputs_ply(config) 
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
if config['emitter'] == 'bayer':
    bayer = dr.zeros(mi.TensorXf, (32, 32, 3))
    bayer[ ::2,  ::2, 2] = 2.2
    bayer[ ::2, 1::2, 1] = 2.2
    bayer[1::2, 1::2, 0] = 2.2

    emitter = {
        'type':'directionalarea',
        'radiance': {
            'type': 'bitmap',
            'bitmap': mi.Bitmap(bayer),
            'raw': True,
            'filter_type': 'nearest'
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

def update_visualization(it, current_img, current_heightmap, reference_image, cbar,
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
    

    img_np = current_img.numpy()
    ref_np = reference_image.numpy()
    error_map = np.abs(img_np - ref_np)
    error_intensity = np.mean(error_map, axis=-1)

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



















#visualize results
plt.ion() # Interactive mode on
fig_vis, axes_vis = plt.subplots(2, 2, figsize=(12, 10))
plt.tight_layout(pad=3.0)
fig_vis.canvas.setWindowTitle('Live Caustic Optimization & Tool Check')

def update_visualization_search(it, current_img, reference_image, params):
    """
    Updates the 2x2 grid with live render, heightmap, and red danger zones.
    """
    
    # Convert Dr.Jit to Numpy
    img_np = current_img.numpy()



    # --- PLOTTING ---
    ax = axes_vis.ravel()
    
    # 1. Top Left: Current Render
    ax[0].clear()
    # Gamma correction (power 0.45) makes the faint caustics visible
    ax[0].imshow(np.clip(img_np ** 0.45, 0, 1)) 
    ax[0].set_title(f"Iter {it}: Render View")
    ax[0].axis('off')
    



    # 1. Setup the lens parameters

    raw_v = params['focused-emitter-shape.vertex_positions'] # This is the 786432 array

    num_vertices = len(raw_v) // 3
    idx = dr.arange(mi.UInt, num_vertices)

    base_x = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 0))
    base_y = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 1))
    base_z = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 2))

    plottingPositions = mi.Vector3f(base_x, base_y, base_z)

    raw_v = params['receiving-plane.vertex_positions'] # This is the 786432 array

    num_vertices = len(raw_v) // 3
    idx = dr.arange(mi.UInt, num_vertices)

    base_x = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 0))
    base_y = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 1))
    base_z = dr.detach(dr.gather(mi.Float, raw_v, idx * 3 + 2))

    ImageplottingPositions = mi.Vector3f(base_x, base_y, base_z)


    #raw_v = params['sensor.to_world'] # This is the 786432 array


    emitterX = (plottingPositions.x)
    emitterY = (plottingPositions.y)

    imageX = ImageplottingPositions.x
    imageY = ImageplottingPositions.y

    mirrorX = [-1, 1]
    MirrorY = [0, 0 ]

    ax[3].clear()  
    ax[3].plot(emitterX, emitterY)
    ax[3].plot(imageX, imageY)
    ax[3].plot(mirrorX, MirrorY)
    


    img_np = current_img.numpy()
    ref_np = reference_image.numpy()
    error_map = np.abs(img_np - ref_np)
    error_intensity = np.mean(error_map, axis=-1)

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











class optimizationObject():

    def __init__(self):
        self.data = np.empty((0, 3))

    def add_point(self, x, y, z):
        new_point = np.array([[x, y, z]])
        self.data = np.append(self.data, new_point, axis=0)




def coarse_alignment_search(x, mapCoords):
    """
    x is a list/array of values proposed by Scipy: [sx, sy, sz, siy]
    """
    # 1. Feed Scipy's guesses into your Mitsuba function
    # We cast to float because Scipy uses float64 and Mitsuba wants float32
    apply_alignment_and_displacement(
        (x[0]), 50, 0, (x[1])
    )

    # 2. Render a very low-res version for speed (64x64 or 128x128)
    # We use dr.detach to make sure we don't accidentally start a gradient graph
    
    image = mi.render(scene, params2, spp=4, spp_grad=(4))

    #update_visualization_search(1, image, image_ref, params2)
    # 3. Calculate a simple L2 loss
    # It's helpful to blur the images here so Scipy "senses" the target easier
    if(dr.sum(image)) == 0: #there is no light on sensor
        return 999999
    else:
        diff = dr.ravel(image) - dr.ravel(image_ref)
        loss = optimizer.energy_stable_concentration_loss(image, image_ref)
#def gaussian_blur_2d(tensor, sigma):
    mapCoords.add_point(float(x[0]) , float(x[1]), float(np.mean(loss.numpy())))
    #update_visualization_search(0, image, image_ref)
    print(f"Testing: {x} | Loss: {loss}")
    return (float(np.mean(loss.numpy())))



#align mesh to the lowest loss before optimizing
def apply_alignment_and_displacement(sx, sy, sz, siy):

    translatedParams = FocusTranslator.translateEmitterAndImage(sx, sy, sz, siy)
    #source
    params2['focused-emitter-shape.vertex_positions'] = translatedParams['focused-emitter-shape.vertex_positions']
    params2['focused-emitter-shape.vertex_normals'] = translatedParams['focused-emitter-shape.vertex_normals']
    #image
    params2['receiving-plane.vertex_positions'] = translatedParams['receiving-plane.vertex_positions']
    params2['receiving-plane.vertex_normals'] = translatedParams['receiving-plane.vertex_normals']
    #sensor
    params2['sensor.to_world'] = translatedParams['sensor.to_world']

    params2.update()




if(align_mesh_to_ref):
    params2 = mi.traverse(scene)

    image = mi.render(scene, params2, spp=4, spp_grad=(4)) # Use slightly higher SPP for alignment
    
    wait = input("Enter to start optimization")


####### Quick Coarse Search#####
    bounds = [(0.1, 60), (0.1, 20)]
    print("[+] Starting Global Search...")
    # shgo is great because it samples the space efficiently

    optimizationMap = optimizationObject()
    optimization_args = [optimizationMap]
    res = shgo(coarse_alignment_search, bounds, iters=50, args=(optimization_args))

    shift_x = mi.Float(res.x[0])
    shift_z = mi.Float(50)
    shift_y = mi.Float(0)
    shift_i_y = mi.Float(res.x[1])

    dr.enable_grad(shift_x, shift_y, shift_z, shift_i_y)

    align_opt = mi.ad.Adam(lr=0.0005)

    print(f"Found Best Start: {res.x}")
    align_opt['sx'] = shift_x
    align_opt['sy'] = shift_z
    align_opt['sz'] = shift_y
    align_opt['siy'] = shift_i_y

    apply_alignment_and_displacement(align_opt['sx'], align_opt['sy'], align_opt['sz'], align_opt['siy'])
    image = mi.render(scene, params2, spp=4, spp_grad=(4)) # Use slightly higher SPP for alignment
    update_visualization_search(1, image, image_ref, params2)
    print(f"  Align | Shift: [{align_opt['sx'][0]:.4f}, {align_opt['sy'][0]:.4f}, {align_opt['sz'][0]:.4f}] Img Shift: {align_opt['siy']}")

    wait = input("run graphing")
#################

    x_Max = optimizationMap.data[:,0].max()
    y_Max = optimizationMap.data[:,1].max()
    x_Min = optimizationMap.data[:,0].min()
    y_Min = optimizationMap.data[:,1].min()



    x_coords = np.linspace(x_Min, x_Max, 100)
    y_coords = np.linspace(y_Min, y_Max, 100)
    X, Y = np.meshgrid(x_coords, y_coords)


    z_values = optimizationMap.data[:, 2]

    q1, q3 = np.percentile(z_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    mask = (z_values >= lower_bound) & (z_values <= upper_bound)
    filtered_data = optimizationMap.data[mask]
    
    points = filtered_data[:, :2] # Gets both X and Y columns
    values = filtered_data[:, 2]  # Gets the Z column
    Z_grid = griddata(points, values, (X, Y), method='linear')


    # 3. Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # cmap='viridis' helps visualize depth
    surf = ax.plot_surface(X, Y, Z_grid, cmap='viridis', edgecolor='none', alpha=0.8)    

    ax.set_xlabel('Emitter Shift X (sx)')
    ax.set_ylabel('Image Plane Shift (siy)')
    ax.set_zlabel('L2 Loss')
    ax.set_title('Caustic Optimization Landscape')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()








    apply_alignment_and_displacement(align_opt['sx'], align_opt['sy'], align_opt['sz'], align_opt['siy'])
    print("[i] Phase 1: Aligning mesh position...")
    # 1. Render and Loss
    image = mi.render(scene, params2, spp=4, spp_grad=(4)) # Use slightly higher SPP for alignment
    

    update_visualization_search(1, image, image_ref, params2)
    
    print(f"  Shift: [{align_opt['sx'][0]:.4f}, {align_opt['sy'][0]:.4f}, {align_opt['sz'][0]:.4f}] Img Shift: {align_opt['siy']}")

    # Capture the final best shift
    final_sx = dr.detach(align_opt['sx'])
    final_sz = dr.detach(align_opt['sz'])
    final_siy = dr.detach(align_opt['siy'])
    print(f"[+] Alignment Frozen at: X={final_sx[0]:.4f}, Z={final_sz[0]:.4f} IMG={final_siy[0]:.4f}")



    wait = input("Enter to continue")























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





def updateScene():
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







































start_time = time.time()
mi.set_log_level(mi.LogLevel.Warn)
iterations = config['max_iterations']
loss_values = []
spp = config['spp']

for it in range(iterations):

    #emitterOrigin=[40, 50, 0]
    

    incidentProportion =  (emitterOrigin[0] / emitterOrigin[1])
    
    #emitterOrigin[0] = emitterOrigin[0] - 1 #increment in X 
    imgLocation = imgLocation + 1  #increment in Z

    imgPosX = (imgLocation * -incidentProportion)
    imgPosZ = imgLocation


    imageTarget=[imgPosX, 0, 0]
    imageOrigin=[imgPosX, imgPosZ, 0]

    sensorPosX = imgLocation * -incidentProportion
    sensorPosZ = imgLocation - 5 #at 20deg fov shift sensor back to be in perfect focus #need to figure out X placement
    sensorTarget=[sensorPosX, 500, 0]
    sensorOrigin=[sensorPosX, sensorPosZ, 0]
    cameraFov=20

    print(f"Image location {imageOrigin}")
    print(f"sensor location {sensorOrigin}")


    resx, resy = config['render_resolution']
    sensor = {
        'type': 'perspective',
        'near_clip': 1,
        'far_clip': 2000,
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


    updateScene = {
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

    scene = mi.load_dict(updateScene)

    # Perform a differentiable rendering of the scene
    image = mi.render(scene, params, seed=it, spp=2 * spp, spp_grad=spp)

    # Scale-independent L2 function
    Z = params['data'] #heightmap tensor

    

    update_visualization(
        it=it,
        current_img=dr.detach(image),           # Detach from graph
        current_heightmap=dr.detach(params['data']), # Detach from graph
        reference_image=image_ref,
        cbar=False,
        aperture_mm=clear_aperture,    
        tool_deg=clearanceAngle,       # Safety margin (use slightly less than 12)
        tool_rad_mm=toolRadius      
    )
    print("press enter to step")
    print(f"Focus at {imgLocation}")
    nothing = input() # i forgot how to do this dont judge







update_visualization(
    it=it,
    current_img=dr.detach(image),           # Detach from graph
    current_heightmap=dr.detach(params['data']), # Detach from graph
    reference_image=image_ref,
    cbar=True,
    aperture_mm=clear_aperture,    
    tool_deg=clearanceAngle,       # Safety margin (use slightly less than 12)
    tool_rad_mm=toolRadius         
)











plt.ioff() # Turn off interactive mode
plt.show() # Keep the window open until you close it


