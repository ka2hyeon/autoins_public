import numpy as np

def compute_dist_mat(adj_mat, threshold = 0):
    assert adj_mat.shape[0] == adj_mat.shape[1]
    dim = adj_mat.shape[0]
    
    A = np.copy(adj_mat)
    dist_mat = np.ones_like(adj_mat)
    for i in range(dim):
        dist_mat[i,i] = 0

    for _ in range(dim-1):
        dist_mat[np.where(A<=threshold)] += 1
        A += np.matmul(A, adj_mat)
    return dist_mat


def to_onehot(x, depth):
    onehot = np.eye(depth)[x]
    return onehot