import matplotlib.pyplot as plt
import numpy as np

from autoins.common import common, io

class RewardShapingTester():
    def __init__(self,
            env,
            exp_dir,
            ccl_id,
            data_name):
        self.env = env
        self.exp_dir = exp_dir
        self.ccl_id = ccl_id
        self.data_name = data_name

        self.io_manager = io.IoManager(exp_dir = exp_dir,
                                        data_name = data_name,
                                        ccl_id = ccl_id)

        self.fig_dir = f'{exp_dir}/figure/{data_name}/{ccl_id}/reward'
        common.create_dir(self.fig_dir, clear_dir = True)

    def test_reward_shaping(self):
        ag_demo_list = self.io_manager.ag_demo
        
        for i, ag_demo in enumerate(ag_demo_list):
            goal = np.expand_dims(ag_demo[-1],0)
            goal = np.tile(goal, [ag_demo.shape[0], 1])

            phi_c = self.env.compute_phi_c(ag_demo, goal)
            phi_g = self.env.compute_phi_g(ag_demo, goal)
            node = self.env.get_node(ag_demo)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(phi_c, label = 'phi_c')
            ax.plot(phi_g, label = 'phi_g')
            ax.legend()
            fig.savefig(f'{self.fig_dir}/reward_shaping_{i:04d}.png')
            plt.close(fig)
        
            #'''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(node)
            ax.legend()
            fig.savefig(f'{self.fig_dir}/node_{i:04d}.png')
            plt.close(fig)
            #'''



        