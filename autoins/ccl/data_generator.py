import numpy as np
from autoins.common import io, math



class DataGenerator():
    def __init__(self,
                    distance_model,
                    exp_dir,
                    ccl_id,
                    data_name,
                    ag_shape,
                    nb_subgoal,
                    nb_data_gen,
                    contrastive_sample_size):

        # input_shape --> ag_shape
        # nb_ccl_sample --> nb_data_gen_size
        # nb_contrastive_sample --> contrastive_sample_size

        self.distance_model = distance_model
        self.exp_dir = exp_dir
        self.ccl_id = ccl_id
        self.data_name = data_name
        self.ag_shape = ag_shape
        self.nb_subgoal = nb_subgoal
        self.nb_data_gen = nb_data_gen
        self.contrastive_sample_size = contrastive_sample_size

        self.io_manager = io.IoManager(self.exp_dir, self.data_name, self.ccl_id)

    def generate_data(self):
        ag_demo = self.io_manager.ag_demo
        adj_mat = self.io_manager.adj_mat
        label = self.io_manager.label
        dist_mat = math.compute_dist_mat(adj_mat)

        contrastive_data = self.generate_contrastive_data(
                                ag_demo,
                                label,
                                dist_mat,
                                contrastive_sample_size = self.contrastive_sample_size,
                                nb_data_gen = self.nb_data_gen)

        classification_data = self.generate_classification_data(
                                ag_demo,
                                label,
                                nb_data_gen = self.nb_data_gen)

        data = dict()
        data.update(contrastive_data)
        data.update(classification_data)
        return data


    def generate_classification_data(self, ag_demo, label, nb_data_gen):
        ag_demo_concat = np.concatenate(ag_demo,0)
        label_concat = np.concatenate(label, 0)

        total_data = len(ag_demo_concat)
        random_idx = np.random.choice(total_data, nb_data_gen, replace = True)

        picked_ag = ag_demo_concat[random_idx]
        picked_label = label_concat[random_idx]
        picked_label_onehot = math.to_onehot(picked_label, depth = self.nb_subgoal)
        data = dict(x_classification = picked_ag, label = picked_label_onehot)
        return data

    def generate_contrastive_data(self, 
                                    ag_demo, 
                                    label, 
                                    dist_mat,
                                    contrastive_sample_size,
                                    nb_data_gen):
        
        positive_pair_array = []
        positive_dist_array = []
        negative_pair_array = []
        negative_dist_array = []

        for _ in range(nb_data_gen):
            positive_pair, positive_dist, negative_pair, negative_dist = \
                self.make_contrastive_sample(ag_demo, label, dist_mat, contrastive_sample_size)

            positive_pair_array.append(positive_pair)
            positive_dist_array.append(positive_dist)
            negative_pair_array.append(negative_pair)
            negative_dist_array.append(negative_dist)

        positive_pair_array = np.asarray(positive_pair_array)
        positive_dist_array = np.asarray(positive_dist_array)
        negative_pair_array = np.asarray(negative_pair_array)
        negative_dist_array = np.asarray(negative_dist_array)

        data = dict(positive_pair = positive_pair_array,
                    positive_dist = positive_dist_array,
                    negative_pair = negative_pair_array,
                    negative_dist = negative_dist_array)
        return data

    def make_contrastive_sample(self, 
                                    ag_demo, 
                                    label, 
                                    dist_mat, 
                                    contrastive_sample_size):
        nb_demo = len(ag_demo)

        demo_idx = np.random.choice(nb_demo, 1, replace = False)[0]
        picked_demo = ag_demo[demo_idx]
        len_demo = picked_demo.shape[0]
                
        t_idx = np.random.choice(len_demo, contrastive_sample_size+1, replace = True)
        picked_sample = picked_demo[t_idx]
        picked_label = label[demo_idx][t_idx]

        anchor = np.repeat(picked_sample[:1,:], contrastive_sample_size, axis = 0)
        query = picked_sample[1:,:]
        
        anchor_t = np.tile(t_idx[:1], [contrastive_sample_size])
        query_t = t_idx[1:] 
        anchor_l = np.repeat(picked_label[:1], contrastive_sample_size, axis = 0)
        query_l = picked_label[1:]
        

        d_for = self._get_graph_dist(anchor_l, query_l, dist_mat) \
                + self._get_temporal_dist(anchor_t, query_t, len_demo)
        d_rev = self._get_graph_dist(query_l, anchor_l, dist_mat) \
                + self._get_temporal_dist(query_t, anchor_t, len_demo)
        d_concat = np.concatenate([d_for, d_rev], 0)

        # pick the index having the minimum distance
        argmin_idx_list = np.where(d_concat == d_concat.min())[0]
        argmin_idx = argmin_idx_list[np.random.choice(argmin_idx_list.shape[0])]

        if argmin_idx < len(d_for):
            idx = argmin_idx
            positive_query = query[[idx]]
            positive_dist = d_for[[idx]] 
            
            positive_pair = np.concatenate([anchor[[0]], positive_query], 0)
            positive_pair = np.expand_dims(positive_pair, 0)
            
            negative_query_for = np.concatenate([query[:idx],query[idx+1:]],0) 
            negative_query_for = np.expand_dims(negative_query_for, 1) 

            negative_query_rev = np.copy(query)
            negative_query_rev = np.expand_dims(negative_query_rev,1)
            
            negative_dist_for = np.concatenate([d_for[:idx],d_for[idx+1:]],0)
            negative_dist_rev = np.copy(d_rev)

        elif argmin_idx >= len(d_for):
            idx = argmin_idx - len(d_for)

            positive_query = query[[idx]]  
            positive_dist = d_rev[[idx]] 

            positive_pair = np.concatenate([positive_query, anchor[[0]]], 0) 
            positive_pair = np.expand_dims(positive_pair, 0)
            
            negative_query_for = np.copy(query)
            negative_query_for = np.expand_dims(negative_query_for, 1) 

            negative_query_rev = np.concatenate([query[:idx],query[idx+1:]],0) 
            negative_query_rev = np.expand_dims(negative_query_rev,1) 

            negative_dist_for = np.copy(d_for) 
            negative_dist_rev = np.concatenate([d_rev[:idx],d_rev[idx+1:]],0) 


        len_for = len(negative_query_for) 
        len_rev = len(negative_query_rev) 
        
        anchor_for = np.expand_dims(anchor[:len_for], 1) 
        anchor_rev = np.expand_dims(anchor[:len_rev], 1) 
        
        negative_pair_for = np.concatenate([anchor_for, negative_query_for], 1) 
        negative_pair_rev = np.concatenate([negative_query_rev, anchor_rev], 1) 
        
        negative_pair = np.concatenate([negative_pair_for, negative_pair_rev], 0)
        negative_dist = np.concatenate([negative_dist_for, negative_dist_rev], 0)         
        return positive_pair, positive_dist, negative_pair, negative_dist

    def _get_graph_dist(self, l0, l1, dist_mat):
        dim = dist_mat.shape[0]
        delta_t = dist_mat[l0, l1]/dim # Normalize from 0 to 1
        return self.distance_model(delta_t)

    def _get_temporal_dist(self, t0, t1, len_demo):
        delta_t = (t1-t0)/len_demo
        return self.distance_model(delta_t)



