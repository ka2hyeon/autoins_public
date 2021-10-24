import os
import pickle
import shutil
import yaml

def load_yaml(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def create_dir(file_dir, clear_dir = False, verbose = 1):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    else:
        if clear_dir == True:
            shutil.rmtree(file_dir)
            os.makedirs(file_dir)
        elif clear_dir == False:
            pass

    if verbose == 0:
        pass
    elif verbose == 1:
        print(f'create {file_dir} (clear_dir = {clear_dir})')