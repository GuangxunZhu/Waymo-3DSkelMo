import os

import numpy as np
import yaml


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Arguments:
    def __init__(self, config_path, filename='default.yaml'):
        with open(os.path.join(config_path, 'smpl.yaml'), 'r') as f:
            smpl = yaml.safe_load(f)
        self.smpl = Struct(**smpl)
        self.smpl.offsets['male'] = np.array(self.smpl.offsets['male'])
        self.smpl.offsets['female'] = np.array(self.smpl.offsets['female'])
        self.smpl.parents = np.array(self.smpl.parents).astype(np.int32)
        self.smpl.joint_num = len(self.smpl.joints_to_use)
        self.smpl.joints_to_use = np.array(self.smpl.joints_to_use)
        self.smpl.joints_to_use = np.arange(0, 156).reshape((-1, 3))[self.smpl.joints_to_use].reshape(-1)

        with open(os.path.join(config_path, 'dog_skel.yaml'), 'r') as f:
            animal = yaml.safe_load(f)
        self.animal = Struct(**animal)
        self.animal.offsets = np.array(self.animal.offsets)
        self.animal.parents = np.array(self.animal.parents).astype(np.int32)
        self.animal.joint_num = len(self.animal.parents)

        self.filename = os.path.splitext(filename)[0]
        with open(os.path.join(config_path, filename), 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, Struct(**value))
            else:
                setattr(self, key, value)

        self.json = config
