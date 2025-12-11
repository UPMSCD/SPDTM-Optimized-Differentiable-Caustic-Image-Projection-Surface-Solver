import os
from os.path import realpath, join
import mirrorMeshes
import drjit as dr


class create:
    def __init__(self, name):

            ###choosing config and scene
        SCENE_DIR = realpath('C:/Users/Conman569/Documents/Notes/Optics/Gregorian Design/NormalizedMetrics/Scripts/causticprojection/caustic/scenes')

        CONFIGS = {
            'wave': {
                'emitter': 'gray',
                'reference': join(SCENE_DIR, 'references/wave-1024.jpg'),
            },
            'sunday': {
                'emitter': 'bayer',
                'reference': join(SCENE_DIR, 'references/sunday-512.jpg'),
            },
                'airforce': {
                'emitter': 'gray',
                'reference': join(SCENE_DIR, 'references/airForce.jpg'),
            },
                'darkside': {
                'emitter': 'gray',
                'reference': join(SCENE_DIR, 'references/darkside.jpg'),
            },
                'star': {
                'emitter': 'gray',
                'reference': join(SCENE_DIR, 'references/star.jpg'),
            },
                'pimpin': {
                'emitter': 'gray',
                'reference': join(SCENE_DIR, 'references/pimpin.jpg'),
            },
                'star3': {
                'emitter': 'gray',
                'reference': join(SCENE_DIR, 'references/star3.jpg'),
            },
        }

        # Pick one of the available configs
        #config_name = 'sunday'


        self.config = CONFIGS[name]
        self.SCENE_DIR = SCENE_DIR
        self.config_name = name

        print('[i] Reference image selected:', self.config['reference'])


    def Outputs(self, config):
        #####output configs
        self.config = config

        self.output_dir = realpath(join('.', 'outputs', self.config_name))
        os.makedirs(self.output_dir, exist_ok=True)
        print('[i] Results will be saved to:', self.output_dir)



        lens_res = self.config.get('lens_res', self.config['heightmap_resolution'])
        self.lens_fname = join(self.output_dir, 'lens_{}_{}.ply'.format(*lens_res))

        if not os.path.isfile(self.lens_fname):
            m = mirrorMeshes.create_flat_lens_mesh(lens_res)
            m.write_ply(self.lens_fname)
            print('[+] Wrote lens mesh ({}x{} tesselation) file to: {}'.format(*lens_res, self.lens_fname))

            








