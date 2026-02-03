







def translateEmitterAndImage(sx, sy, sz, siy):

    import os
    from os.path import realpath, join

    import drjit as dr
    import mitsuba as mi
    import time
    import startupConfig
    import mirrorMeshes
    import numpy as np

    

    sx = float(mi.Float(sx).numpy())
    sy = float(mi.Float(sy).numpy())
    sz = float(mi.Float(sz).numpy())
    siy = float(mi.Float(siy).numpy())


    #scene parameters as inputs
    renderRes = (256, 256)
    heihgtmapRes = (512, 512)
    optimizationSteps = 100
    meshSaampling = 64



    mirrorTarget=[0,1,0]
    mirrorOrigin=[0,0,0]

    emitterTarget=[0, 0, 0]
    emitterOrigin=[sx, sy, sz]


    import math
    #need to use atan to get angle incident

    incidentProportion =  (emitterOrigin[0] / emitterOrigin[1])
    rads = math.atan(incidentProportion)
    degs = math.degrees(rads)

    imgLocation = siy

    imgPosX = (imgLocation * -incidentProportion)
    imgPosZ = imgLocation
    imageTarget=[0, 0, 0] #################Change to imgPosX to be parallel to mirror
    imageOrigin=[imgPosX, imgPosZ, 0]

    sensorPosX = imgLocation * -incidentProportion
    sensorPosZ = imgLocation - 5 #at 20deg fov shift sensor back to be in perfect focus #need to figure out X placement
    sensorTarget=[sensorPosX, 500, 0]
    sensorOrigin=[sensorPosX, sensorPosZ, 0]
    cameraFov=20









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
            'width': 512,
            'height': 512,
            'pixel_format': 'rgb',
            'rfilter': {
                # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
                'type': 'gaussian'
            }
        },
    }


    translatedScene = {
        'type': 'scene',
        'sensor': sensor,
        'integrator': integrator,
        # Glass BSDF
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


    

    translatedScene = mi.load_dict(translatedScene)

    translatedParams = mi.traverse(translatedScene)

    return translatedParams
