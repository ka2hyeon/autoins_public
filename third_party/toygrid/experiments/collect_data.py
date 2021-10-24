# python -m experiments.collect_data

import numpy as np
from toygrid import ToyGridEnv, ToyGridDemonstrator

SCALE = 2.5
def convert_ag(p):
    '''
    9by9 --> 36 by 36
    '''
    achieved_goal = []
    for p in path:
        ag = np.asarray([SCALE*(p[0]-1),
                        -SCALE*(p[1]-1)])
        achieved_goal.append(ag)
    return np.asarray(achieved_goal)


if __name__ == '__main__':
    nb_demo = 50
    save_dir = './experiments/result'
    env = ToyGridEnv()
    demonstrator = ToyGridDemonstrator(env)

    obs_list = []
    ag_list = []
    demo_label_list = []
    for _ in range(nb_demo):
        env.reset()
        path = demonstrator.demonstrate()
        if path is not None:
            obs_list.append(np.asarray(path))
            ag_list.append(convert_ag(path))
            demo_label_list.append('target1')


    env.render_path(path_list = obs_list, save_path = f'{save_dir}/toygrid.png')
    np.save(f'{save_dir}/obs.npy', obs_list)
    np.save(f'{save_dir}/ag.npy', ag_list)
    np.save(f'{save_dir}/demo_label.npy', demo_label_list)