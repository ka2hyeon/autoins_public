from autoins.common import math, io

class GraphModel():
    def __init__(self,
                    exp_dir,
                    ccl_id,
                    data_name):
        self.exp_dir = exp_dir
        self.ccl_id = ccl_id
        self.data_name = data_name

        self.io_manager = io.IoManager(exp_dir = self.exp_dir,
                                        data_name =  self.data_name,
                                        ccl_id =  self.ccl_id)
        self.build()
        
    def build(self):
        adj_mat = self.io_manager.adj_mat 
        self._dist_mat = math.compute_dist_mat(adj_mat)

    @property
    def dist_mat(self):
        return self._dist_mat