import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from autoins.common import common, io

class CclTester():
    def __init__(self, 
                    ccl_model,
                    exp_dir,
                    ccl_id,
                    data_name):

        self.ccl_model = ccl_model
        self.exp_dir = exp_dir
        self.data_name = data_name
        self.ccl_id = ccl_id
        
        self.io_manager = io.IoManager(exp_dir = self.exp_dir,
                                        data_name =  self.data_name,
                                        ccl_id =  self.ccl_id)

        self.fig_dir = f'{exp_dir}/figure/{data_name}/{ccl_id}/test'
        common.create_dir(self.fig_dir, clear_dir = True)
        
    def plot_ag_on_env(self):
        assert self.data_name == 'ant'
        from toygrid import ToyGridEnv
        SCALE = 2.5

        env = ToyGridEnv()
        ag_demo = self.io_manager.ag_demo
        label = self.io_manager.label

        state_demo = []           
        for ag_traj in ag_demo:
            state_traj = np.copy(ag_traj)
            state_traj[:,0] = (state_traj[:,0]/SCALE)+1
            state_traj[:,1] = (state_traj[:,1]/-SCALE)+1
            state_demo.append(state_traj)

        env.render_path(path_list = state_demo,
                        label_list = label,
                        save_path = f'{self.fig_dir}/plot_ag_on_env.png')


    def plot_adj_mat(self):
        A = self.io_manager.adj_mat
        dim = A.shape[0]
    
        G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.DiGraph)
        layout = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')

        color = []
        for i in range(dim):
            color.append('C%d'%i)

        fig = plt.figure(figsize = (4,4))
        ax = fig.add_subplot(111)
        nx.draw(G, node_size = 100, pos = layout, ax = ax, 
                    node_color = color, with_labels=True )
        
        plt.savefig(f'{self.fig_dir}/dag.png')
        plt.close(fig)
