import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import scipy
from scipy.optimize import minimize

from autoins.common import common, io

class AdjacentMatrixComputer():
    def __call__(self, label_demo, nb_subgoal):

        ## (1) Count frequency
        node_frequency = np.zeros(nb_subgoal)
        for label in label_demo:
            for l in label:
                node_frequency[l] += 1
        
        ## (2) Count joint frequency
        joint_frequency = np.zeros([nb_subgoal, nb_subgoal])
        for label in label_demo:
            for l0, l1 in zip(label[:-1], label[1:]):
                joint_frequency[l0, l1] += 1

        ## (3) Compute adjacent matrix
        adj_mat = np.zeros([nb_subgoal, nb_subgoal])
        for i in range(nb_subgoal):
            if node_frequency[i] == 0:
                adj_mat[i,:] = 0
            else:
                adj_mat[i,:] = joint_frequency[i,:]/node_frequency[i]
        return adj_mat

    def plot(self):
        pass

class BeamSearcher():
    def __call__(self, p_traj, A, K=30, nb_cut_off = 5, fig_dir = None, verbose = 0):
        assert A.shape[0] == A.shape[1] == p_traj.shape[1]
        T_max, N = p_traj.shape
        
        A_aug = A.copy()
        #for d in range(1): ###### N
        #    A_aug += np.matmul(A,A_aug)
        A_aug  += np.eye(N)
        log_p_traj = np.log(p_traj+1e-23)

        ## abstract p_traj 
        abstract_log_p_traj = []; mapping = []
        for t in range(T_max):
            log_p = log_p_traj[t,:].copy()
            sorted_idx = np.argsort(log_p)[::-1][:nb_cut_off]  

            if t ==0:
                abstract_log_p_traj.append(log_p)
                sorted_idx_prev = sorted_idx.copy()
            else:
                if np.array_equal(sorted_idx_prev, sorted_idx):
                    abstract_log_p_traj[-1] += log_p
                else:
                    abstract_log_p_traj.append(log_p)
                sorted_idx_prev =  sorted_idx.copy()
            mapping.append(len(abstract_log_p_traj)-1)
        abstract_log_p_traj = np.asarray(abstract_log_p_traj)
        abstract_T_max = abstract_log_p_traj.shape[0]
        
        mapping_reversed = []
        start = 0
        prev_abstract_t = mapping[start]
        for t in range(T_max):
            abstract_t = mapping[t]
            if abstract_t != prev_abstract_t:
                end = t-1
                mapping_reversed.append([start, end])
                start = t
            prev_abstract_t = abstract_t
        end = T_max-1
        mapping_reversed.append([start, end])

        ## beamsearch
        sequences = [[list(), 0, []]]
        if fig_dir:
            vis_traj = []
        
        for t in range(abstract_T_max):
            log_p = abstract_log_p_traj[t,:]
            all_candidates = list()
            
            for i in range(len(sequences)):
                seq, score, visited = sequences[i]
                for j in range(N):
                    if not seq:
                        candidate = [seq+[j], score+log_p[j], visited]
                        all_candidates.append(candidate)
                    else:
                        last_seq = seq[-1]
                        if (j not in visited) and (A_aug[last_seq, j]>0): 
                            if last_seq == j:
                                candidate = [seq+[j], score + log_p[j], visited]
                            else:
                                candidate = [seq+[j], score + log_p[j], visited+[last_seq]]
                            all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse = True)
            sequences = ordered[:K]
            if fig_dir:
                sequences_for_vis = []
                for seq in sequences:
                    abs_traj = seq[0]
                    abs_p = seq[1]
                    abs_visited = seq[2]

                    traj = []
                    for abs_t, abs_value in enumerate(abs_traj):
                        start, end = mapping_reversed[abs_t]
                        traj += ([abs_value]*int(end-start+1))
                    
                    sequences_for_vis.append([traj, abs_p, abs_visited])  
                vis_traj.append(sequences_for_vis)
        
        
        ## decode abstracted p traj
        abstract_max_sequence = sequences[0]

        max_sequence_traj = []
        abstract_max_sequence_traj = abstract_max_sequence[0]
        abstract_max_value = abstract_max_sequence[1]
        abstract_max_visited = abstract_max_sequence[2]
        for t in range(T_max):
            abstract_t = mapping[t]
            max_sequence_traj.append(abstract_max_sequence_traj[abstract_t])
        
        max_sequence = [max_sequence_traj, abstract_max_value, abstract_max_visited]

        if fig_dir: 
            self.plot(log_p_traj, vis_traj, fig_dir, verbose)          
        return np.asarray(max_sequence[0])

    def plot(self, value_traj, vis_traj, fig_dir, verbose = 0):
        T_max, N = value_traj.shape
        
        if T_max <= 50:
            skipped = 1
            fig_ratio = 1
        else:
            skipped = int(T_max/50)
            fig_ratio = int((T_max/skipped)/N)

        ## create figure
        fig_width = 5*fig_ratio
        fig_heigt = 5
        fig1 = plt.figure(figsize = [fig_width, fig_heigt])
        ax1 = fig1.add_subplot(111)

        ## draw porb image
        value_img = np.transpose(value_traj[::skipped,:])
        sns.heatmap(value_img, ax = ax1, linewidth = 0.3)

        ## draw beamsearch path
        for t, vis in enumerate(vis_traj):
            if verbose == 0:
                if t<len(vis_traj)-1:
                    continue
            elif verbose == 1:
                pass

            for beam in vis:
                # beam: [node_traj, prob. previously visited node]
                node_traj = beam[0][::skipped]
                prob = beam[1]
                visited_node = beam[2]
                
                x_range = np.arange(len(node_traj))+0.5
                y_range = np.asarray(node_traj)+0.5
                ax1.cla()
                ax1.plot(x_range,y_range, color = 'white', linewidth = 2)
                ax1.scatter(x_range,y_range, color = 'black')
            
            fig1.savefig(f"{fig_dir}/{t:04d}.png")
        plt.close(fig1)

class NearestAcyclicMatrixFinder:
    def __init__(self, tol = 1e-3):
        self.tol = tol
    
    """
    def __call__(self, A, loss_type = 'kl', fig_dir = None):
        assert A.shape[0] == A.shape[1]
        dim = A.shape[0]

        A_vec = np.reshape(A, [-1])
        def loss(x):
            return self._loss(x, A_vec, loss_type = loss_type)

        acyclic_constraint = lambda x: self._is_acyclic(x)
        cons = [{'type': 'eq', 'fun': acyclic_constraint}]
        for i in range(dim):
            dist_constraint_i = lambda x, i=i: self._is_dist(x,i=i)
            cons.append({'type': 'eq', 'fun': dist_constraint_i})

        bnds = [(0,None) for _ in range(dim*dim)]
        res = scipy.optimize.minimize(loss, A_vec,
                        method = 'SLSQP',
                        constraints = cons,
                        bounds = tuple(bnds))
        print('acyclicity:', self._is_acyclic(res.x))
        res_x = np.copy(res.x)
        res_x[res_x<=self.tol]=0
        new_A = np.reshape(res_x, A.shape)

        if fig_dir:
            self.plot_op1(A, new_A, fig_dir)
            self.plot_op2(A, new_A, fig_dir)    
        return new_A
    """
    
    def __call__(self, A, fig_dir = None):
        dim = A.shape[0]
        max_search = 10000

        a_idx = np.where(A>0)
        k = np.count_nonzero(A>0)
        assert len(a_idx[0]) == k

        a = A[a_idx]
        a_sorted = np.sort(a)
        a_argsorted = np.argsort(a)


        N = np.minimum(2**k, max_search)
        A_cut_list = []
        acyclicity_list = []
        x = int('0', 2)

        for n in range(int(N)): # np returns float
            idx_list = self._binary_to_idx_list(x)
            #print(idx_list)
            cut_idx = self._get_cut_idx(idx_list, a_idx, a_argsorted)
            A_cut = self._cut_matrix(A, cut_idx)
            acyclicity = self._is_acyclic(A_cut.reshape(-1))
                
            A_cut_list.append(A_cut)
            acyclicity_list.append(acyclicity)
            x += int('1',2)

        argmin = np.argmin(acyclicity_list)
        new_A = A_cut_list[argmin]
        print('acyclicity:', acyclicity_list[argmin])
        #import IPython
        #IPython.embed()
        if fig_dir:
            self.plot_op1(A, new_A, fig_dir)
            self.plot_op2(A, new_A, fig_dir)    
        return new_A

    
    def _pruning(self, A, fig_dir = None):
        assert A.shape[0] == A.shape[1]
        dim = A.shape[0]

        init_max_tol = 1
        init_min_tol = 0
        tol = (init_max_tol+init_min_tol)/2, m

        for i in range(30):        
            prev_tol = tol    
            A_candi = np.copy(A)
            A_candi[A<tol] = 0

            acyclicity = self._is_acyclic(A_candi.reshape(-1)) 
            if acyclicity == 0:
                init_max_tol = tol
            else:
                init_min_tol = tol
            tol = (init_max_tol+init_min_tol)/2
            print(f'{i}th iteration, tol={tol}, acyclicity={acyclicity}')
        new_A = np.copy(A)
        new_A[A<init_max_tol] = 0
        if fig_dir:
            self.plot_op1(A, new_A, fig_dir)
            self.plot_op2(A, new_A, fig_dir)    
        return new_A

    def _is_acyclic(self, x):
        """
        D. Wei (2018, Nips) "DAGs with No Fears ..."
        """
        dim = int(np.sqrt(x.shape[0]))
        A = np.reshape(x, [dim, dim])

        A_tilde = np.copy(A)
        for i in range(dim):
            A_tilde[i,i] = 0

        A_expm = np.eye(dim)
        mul_term = np.eye(dim)
        for _ in range(dim):
            mul_term = np.matmul(mul_term, A_tilde)
            A_expm += mul_term
        A_trace = np.trace(A_expm)
        return np.linalg.norm(A_trace-dim)

    def _is_dist(self, x, i):
        dim = int(np.sqrt(x.shape[0]))
        A = np.reshape(x, [dim, dim])
        return np.sum(A[i,:])-1

    def _is_not_isolated(self, x):
        dim = int(np.sqrt(x.shape[0]))
        A = np.reshape(x, [dim, dim])

        sum = []
        for i in range(A):
            sum.append(np.sum(A[i,:]))
        return sum

    def _cut_matrix(self, A, idx):
        new_A = np.copy(A)
        new_A[idx] = 0
        return new_A

    def _get_cut_idx(self, idx_list, a_idx, a_argsorted):
        cut_idx_i = []
        cut_idx_j = []
        for idx in idx_list:
            a_argosrt_idx = a_argsorted[idx]
            cut_idx_i.append(a_idx[0][a_argosrt_idx])
            cut_idx_j.append(a_idx[1][a_argosrt_idx])
        return (np.asarray(cut_idx_i, dtype = np.int32), 
                np.asarray(cut_idx_j, dtype = np.int32))

    def _binary_to_idx_list(self, x):
        # x: binary
        max_digit = int(np.log2( np.maximum(x,1)))+1
        
        idx_list = []
        for k in range(1, max_digit+1):
            extractor = int('1', 2) << (k-1)
            bit =  (x & extractor) >> (k-1)
            is_one = (bit == int('1',2))
            #print(f'digit:{k}, bit:{bin(bit)}, is_one: {is_one}')
            if is_one:
                idx_list.append(k-1)
        return idx_list

    def _loss(self, x, y, loss_type):
        dim = int(np.sqrt(x.shape[0]))

        if loss_type == 'l1_norm':
            return np.sum(np.abs(x-y))

        elif loss_type == 'l2_norm':
            return np.sqrt(np.sum(np.square(x-y)))

        elif loss_type == 'kl':
            x_mat = np.reshape(x, [dim, dim])
            y_mat = np.reshape(y, [dim, dim])
            
            kl = 0
            for i in range(dim):
                p = np.maximum(x_mat[i,:], 1e-30)
                q = np.maximum(y_mat[i,:], 1e-30)
                kl += np.sum(p * np.log( p/q))
            return kl

    def plot_op1(self, A, new_A, fig_dir):
        dim = A.shape[0]
        G1 = nx.from_numpy_matrix(np.matrix(A), create_using=nx.DiGraph)
        G2 = nx.from_numpy_matrix(np.matrix(new_A), create_using=nx.DiGraph)
        layout = nx.spring_layout(G2)
        #layout = nx.kamada_kawai_layout(G2)

        color = []
        for i in range(dim):
            color.append('C%d'%i)
        
        fig = plt.figure(figsize = (8,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        nx.draw(G1, node_size = 100, pos = layout, ax = ax1, node_color = color, with_labels=True )
        nx.draw(G2, node_size= 100, pos = layout, ax = ax2, node_color = color, with_labels=True)

        plt.savefig(f'{fig_dir}/dag.png')
        plt.close(fig)

    def plot_op2(self, A, new_A, fig_dir):
        dim = A.shape[0]
        fig = plt.figure(figsize = (8,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        sns.heatmap(A, ax = ax1)
        sns.heatmap(new_A, ax = ax2)
        plt.savefig(f'{fig_dir}/adjacent_matrix.png')
        plt.close(fig)

class CclLabeler():
    def __init__(self,
                ccl_model, 
                exp_dir,
                data_name,
                ccl_id,
                max_beam_search,
                nb_subgoal,
                initialize = False,
                plot_cycle = 10):
        
        self.ccl_model = ccl_model
        self.nb_subgoal = nb_subgoal
        self.max_beam_search = max_beam_search
        self.plot_cycle = plot_cycle
        self.io_manager = io.IoManager(exp_dir = exp_dir,
                                        data_name = data_name,
                                        ccl_id = ccl_id)
        self._count = 0        
        self.fig_dir_beamsearch = f'{exp_dir}/figure/{data_name}/{ccl_id}/beamsearch'
        self.fig_dir_adjmat = f'{exp_dir}/figure/{data_name}/{ccl_id}/adjmat'
        
        if initialize:
            common.create_dir(self.fig_dir_beamsearch, clear_dir = True)
            common.create_dir(self.fig_dir_adjmat, clear_dir = True)
    
    def update_label(self):
        self._count += 1
        ag_demo = self.io_manager.ag_demo

        if self._count == 1:
            new_adj_mat, new_label_demo = self._initialize_label(ag_demo)
        else:
            label_demo, prob_demo = self._predict_label(ag_demo)
            adj_mat = self._compute_adj_mat(label_demo)            
            new_adj_mat = self._find_nearest_dag(adj_mat)
            
            new_label_demo = self._beamsearch(prob_demo, new_adj_mat)
            new_label_demo = self._rectify_isolation_label(new_label_demo, new_adj_mat)
            new_adj_mat = self._compute_adj_mat(new_label_demo)  

        self.io_manager.adj_mat = new_adj_mat
        self.io_manager.label = new_label_demo
        return new_adj_mat, new_label_demo

    def _beamsearch(self, prob_demo, adj_mat):
        beamsearch = BeamSearcher()
        label_demo = []
        for i, prob_traj in enumerate(prob_demo):
            if (self._count % self.plot_cycle == 0) and (i == 0):
                fig_dir = f'{self.fig_dir_beamsearch}/{self._count:06d}'
                common.create_dir(fig_dir, clear_dir = True)
            else:
                fig_dir = None
            
            label_traj = beamsearch(prob_traj, adj_mat, K = self.max_beam_search, fig_dir=fig_dir)
            label_demo.append(label_traj)
        return label_demo

    def _compute_adj_mat(self, label_demo):
        compute_adj_mat = AdjacentMatrixComputer()
        adj_mat = compute_adj_mat(label_demo, self.nb_subgoal)
        return adj_mat

    def _find_nearest_dag(self, adj_mat):
        if self._count % self.plot_cycle == 0:
            fig_dir = f'{self.fig_dir_adjmat}/{self._count:06d}'
            common.create_dir(fig_dir, clear_dir = True)
        else:
            fig_dir = None

        find_nearest_dag = NearestAcyclicMatrixFinder()
        adj_mat_dag = find_nearest_dag(adj_mat, fig_dir = fig_dir)
        return adj_mat_dag

    def _initialize_label(self, ag_demo_list):
        nb_subgoal = self.nb_subgoal
        compute_adj_matrix = AdjacentMatrixComputer()
        
        label_demo = []
        for ag_demo in ag_demo_list:
            len_demo = len(ag_demo)
            quotient, remainder = np.divmod(len_demo, nb_subgoal)

            label = []
            for i in range(nb_subgoal):
                if i <= remainder-1:                    
                    len_partition = quotient + 1
                else:
                    len_partition = quotient
                l = i*np.ones(len_partition) 
                label.append(l.astype(np.int32))
            label = np.concatenate(label,0)
            label_demo.append(label)
        adj_mat = compute_adj_matrix(label_demo, self.nb_subgoal)
        return adj_mat, label_demo

    def _predict_label(self, ag_demo_list):
        label_demo = []
        prob_demo = []
        for ag_demo in ag_demo_list:
            feature = self.ccl_model.feature_model.predict(ag_demo)
            prob = self.ccl_model.classifier_model.predict(feature)
            label = np.argmax(prob, axis = -1)
            prob_demo.append(prob)
            label_demo.append(label)
        return label_demo, prob_demo

    def _rectify_isolation_label(self, labels, A):
        isolation_tol = 0
        nb_subgoal = self.nb_subgoal
        ## find isolated nodes
        isolated_nodes = []
        for i in range(nb_subgoal):
            input_strength =  np.sum(A[i,:])-A[i,i]
            output_strength =  np.sum(A[:,i])-A[i,i]
            if input_strength <= isolation_tol and output_strength <= isolation_tol:
                isolated_nodes.append(i)

        print('nb of isolated nodes:', len(isolated_nodes))

        ## count label frequency
        label_frequency = [0]*nb_subgoal
        for l_traj in labels:
            for l in l_traj:
                label_frequency[l] += 1
        largest_frequency_index = sorted(range(len(label_frequency)), 
                                            key=lambda k: label_frequency[k],
                                            reverse = True)
        
        ## select node to be divided
        nodes_to_be_divided = []
        for idx, isolated_node in enumerate(isolated_nodes):
            largest_node = largest_frequency_index[idx]
            nodes_to_be_divided.append(largest_node)
         
        new_labels = []
        for l_traj in labels:
            new_l_traj = l_traj.copy()
            for isol_node, div_node in zip(isolated_nodes, nodes_to_be_divided):
                candidate = np.where(l_traj == div_node)
                nb_label_to_be_changed = int(len(candidate[0])/2)
                index_to_be_changed = (candidate[0][-nb_label_to_be_changed:],)
                new_l_traj[index_to_be_changed] = isol_node
            new_labels.append(new_l_traj)
        return new_labels

"""
if __name__ == '__main__':
    A = np.asarray([[0.6, 0.4],
                    [0.3, 0.7]])

    A_dag = NearestAcyclicMatrixFinder().optimize(A)
    print(A_dag)
"""