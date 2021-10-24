import matplotlib.pyplot as plt
import numpy as np
from autoins.common import common, io

class SvddTester():
    def __init__(self,
                svdd,
                exp_dir,
                ccl_id,
                data_name):
            
        self.svdd = svdd
        self.exp_dir = exp_dir
        self.ccl_id = ccl_id
        self.data_name = data_name

        self.io_manager = io.IoManager(self.exp_dir, 
                                        self.data_name, 
                                        self.ccl_id)
        
        self.fig_dir = f'{exp_dir}/figure/{data_name}/{ccl_id}/test_svdd'
        common.create_dir(self.fig_dir, clear_dir = True)

    def test_ood_ant(self):
        assert self.data_name == 'ant'
        from toygrid import ToyGridEnv
        SCALE = 2.5

        world = ToyGridEnv().world

        ag_demo = self.io_manager.ag_demo
        label = self.io_manager.label
        state_demo = []           
        for ag_traj in ag_demo:
            state_traj = np.copy(ag_traj)
            state_traj[:,0] = (state_traj[:,0]/SCALE)+1
            state_traj[:,1] = (state_traj[:,1]/-SCALE)+1
            state_demo.append(state_traj)
        state_data = np.concatenate(state_demo, 0)
        ag_data = np.concatenate(ag_demo, 0)

        n = 100
        x = np.linspace(-5, 20, n)
        y = np.linspace(-20, 5, n)
        xv, yv = np.meshgrid(x, y)
        xv_state = (xv/SCALE)
        yv_state = (yv/-SCALE)

        xv_reshaped = np.reshape(xv,[-1,1])
        yv_reshaped = np.reshape(yv,[-1,1])
        ag = np.concatenate([xv_reshaped, yv_reshaped], axis = 1)
        
        predicted_c = self.svdd.predict(ag)
        predicted_c = np.reshape(predicted_c, [n,n])

        predicted_c_binary = self.svdd.predict_binary(ag)
        predicted_c_binary = np.reshape(predicted_c_binary, [n,n])

        fig = plt.figure()
        ax = fig.add_subplot(111) 
        plt.pcolormesh(xv, yv, predicted_c)
        fig.savefig(f'{self.fig_dir}/ood.png')
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111) 
        ax.imshow(world)
        ax.scatter(state_data[:,0], state_data[:,1], s = 5, label ='data')
        ax.contourf(xv_state, yv_state, predicted_c_binary, alpha = 0.5)
        ax.legend()

        fig.savefig(f'{self.fig_dir}/ood_binary.png')
        plt.close(fig)
