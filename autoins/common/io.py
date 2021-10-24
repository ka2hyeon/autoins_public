import numpy as np
from autoins.common import common

class IoManager:
    def __init__(self,
                    exp_dir,
                    data_name,
                    ccl_id):
        self.exp_dir = exp_dir
        self.data_name = data_name
        self.ccl_id = ccl_id
        self.initialize_dirs(clear_dir = False)

    def initialize_dirs(self, clear_dir = False):
        self.dir_dict = {}
        self.dir_dict['demo_data'] = f'{self.exp_dir}/data/demo/{self.data_name}'
        self.dir_dict['ccl_data'] = f'{self.exp_dir}/data/ccl/{self.data_name}/{self.ccl_id}'
        self.dir_dict['ccl_weight'] = f'{self.exp_dir}/weight/ccl/{self.data_name}/{self.ccl_id}'
        self.dir_dict['svdd_weight'] = f'{self.exp_dir}/weight/svdd/{self.data_name}/{self.ccl_id}'
        
        for key, dir in self.dir_dict.items():
            if key == 'demo_data':
                common.create_dir(dir, clear_dir = False)
            else:
                common.create_dir(dir, clear_dir = clear_dir)

    @property
    def adj_mat(self):
        _adj_mat = np.load(f"{self.dir_dict['ccl_data']}/adj_mat.npy")
        return _adj_mat


    @property
    def ccl_weight_dir(self):
        return self.dir_dict['ccl_weight']

    @property
    def svdd_weight_dir(self):
        return self.dir_dict['svdd_weight']

    @property
    def label(self):
        _label = np.load(f"{self.dir_dict['ccl_data']}/label.npy", allow_pickle = True)
        return  list(_label)

    @property
    def ag_demo(self):
        _ag = np.load(f"{self.dir_dict['demo_data']}/ag.npy", allow_pickle = True)
        return list(_ag)

    @adj_mat.setter
    def adj_mat(self, value):
        np.save(f"{self.dir_dict['ccl_data']}/adj_mat.npy", value)
    
    @label.setter
    def label(self, value):
        np.save(f"{self.dir_dict['ccl_data']}/label.npy", value)

    
