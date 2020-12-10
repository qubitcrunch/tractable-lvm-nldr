import time
import scipy.sparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dist_helpers import compute_dis_nbr_radii


def compute_knn(X: np.ndarray, n_neighs: int) -> (np.ndarray, np.ndarray):
    """

    Args:
        X: np.ndarray = data matrix of shape (D, N) where D is the raw dimensionality and N is number of points
        n_neighs: int = number of nearest neighbors
        block_size: int
    Returns:

    """
    neigh = NearestNeighbors(n_neighbors=n_neighs+1)
    neigh = neigh.fit(X)
    knn_dist, knn_idx = neigh.kneighbors(X, return_distance=True)
    return knn_idx[:, -n_neighs:], knn_dist[:, -n_neighs:]  # ignore dist and idx to self


def compute_neighborhood_graph(X: np.ndarray, n_neighs: int, steps: int) -> np.ndarray:
    """

    Args:
        X: np.ndarray = data matrix of shape (D, N) where D is the raw dimensionality and N is number of points
        n_neighs: int = number of nearest neighbors
        steps: int = number of steps along the nearest neighbor graph

    Returns:

    """
    n_points = X.shape[1]
    # compute k nearest neighbors
    print(f'Computing k-nearest neighbors with k={n_neighs} ...')
    start_time = time.time()
    knn_idx, knn_dst = compute_knn(X, n_neighs)
    print(f'Time elapsed: {time.time() - start_time}')

    # build graph
    print(f'Building neighborhood graph with k={n_neighs}, s={steps} ...')
    start_time = time.time()
    row_k = np.matlib.repmat(range(n_points), n_neighs, 1)
    Kd = scipy.sparse.csr_matrix((knn_dst, (knn_idx, row_k)), shape=(n_points, n_points))

    # random walk with s steps
    K = Kd.copy().tocsr().fill(1)
    R = K.copy()
    for s in range(1, steps):
        R = R + K ** s

    R = R.copy().tocsr().fill(1)
    R[range(0, n_points + 1, n_points ^ 2)] = 0

    # adjacency matrix for minimum spanning tree
    adjacency = scipy.sparse.csgraph.minimum_spanning_tree(np.max(Kd, Kd.T)).toarray()

    # asymmetric similarity graph (with distances along edges)
    E = Kd.multiply(max(adjacency, R.multiply(R.T)))  # element-wise multiply

    # warn if the graph is disconnected
    G = np.max(E, E.T)
    # TODO: translate this part of the MATLAB code
    print(f'Time elapsed: {time.time() - start_time}')
    return E


def coarse_grain(X: np.ndarray, depth: int, indices: np.ndarray = None, votes: np.ndarray = None) -> dict:
    """

    Args:
        X:
        depth:
        indices:
        votes:

    Returns:

    """
    n_points = X.shape[1]
    if not indices:
        indices = np.array((range(n_points)))

    if not votes:
        votes = np.ones((n_points, 1))

    n_points = len(indices)
    tree = {'depth': depth, 'indices': indices, 'votes': votes}

    # recursion stopping condition
    if depth == 0:
        return tree

    # calculate the landmarks
    n_landmarks = round((0.5*n_points*n_points)**(1/3))
    landmarks = np.random.choice(n_points, n_landmarks, replace=False)
    tree['down'] = [0]*landmarks

    # underlings
    tree['landmark_indices'] = landmarks
    neigh = NearestNeighbors(n_neighbors=1)
    neigh = neigh.fit(X[:, landmarks])
    closest, tree['dist_to_closest_landmark'] = neigh.kneighbors(X[:, indices], return_distance=True)
    tree['indices_closest_landmark'] = landmarks[closest]

    # recurse
    up_votes = np.zeros(n_landmarks)
    for l in range(n_landmarks):
        log_idx = np.where(closest == l)[0]
        sub_idx = indices[log_idx]
        sub_votes = votes[log_idx]
        up_votes[l] = np.sum(sub_votes)
        tree['down'][l] = [coarse_grain(X, depth-1, sub_idx, sub_votes)]

    tree['up'] = coarse_grain(X, depth-1, landmarks, up_votes)
    return tree


def compute_sim_recurse(tree: dict) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """

    Args:
        tree:

    Returns:

    """
    # base case
    if tree['dict'] == 0:
        return [], [], [], []

    # at node
    row_l = tree['indices']
    col_l = tree['indices_closest_landmark']
    w_l = tree['votes'] ** 2
    dst_l = tree['dist_to_closest_landmark']
    not_self = np.where(row_l != col_l)[0]
    row_l = row_l[not_self]
    col_l = col_l[not_self]
    w_l = w_l[not_self]
    dst_l = dst_l[not_self]

    # recurse?
    if tree['depth'] == 1:
        return

    # recurse down
    Nc = len(tree['down'])
    for c in range(Nc):
        (row_c, col_c, w_c, dst_c) = compute_sim_recurse(tree['down'][c])
        row_l = np.vstack((row_l, row_c))
        col_l = np.vstack((col_l, col_c))
        w_l = np.vsplit((w_l, w_c))
        dst_l = np.vstack((dst_l, dst_c))

    # recurse up
    (row_u, col_u, w_u, dst_u) = compute_sim_recurse(tree['up'])
    row_l = np.vstack((row_l, row_u))
    col_l = np.vstack((col_l, col_u))
    w_l = np.vstack((w_l, w_u))
    dst_l = np.vstack((dst_l, dst_u))

    return row_l, col_l, w_l, dst_l


def compute_sim(tree: dict, E: scipy.sparse.csr_matrix) -> dict:
    """

    Args:
        tree:
        E:

    Returns:

    """
    # edges from neighborhood graph
    row_e, col_e = np.nonzero(E)
    dst_e = E[row_e, col_e]

    # weights
    w_e = np.ones(len(row_e))

    # landmarks
    (row_l, col_l, w_l, dst_l) = compute_sim_recurse(tree)

    # normalize
    N = len(tree['indices'])
    w_l = N * (w_l / np.sum(w_l))

    # combine
    row_s = np.vstack((row_e, row_l))
    col_s = np.vstack((col_e, col_l))
    w_s = np.vstack((w_e, w_l))
    w_s = w_s / np.sum(w_s)
    dst_s = np.vstack((dst_e, dst_l))

    # distances to shrink
    delta_sq_s = (dst_s ** 2) / (2 * np.log(2))

    # sort
    hstacked = np.hstack((col_s, row_s))
    ii = np.lexsort(np.fliplr(hstacked).T)

    # dict
    sim_pairs = {}
    sim_pairs['rows'] = row_s[ii]
    sim_pairs['cols'] = col_s[ii]
    sim_pairs['weights'] = w_s[ii]
    sim_pairs['delta_sq'] = float(delta_sq_s[ii])
    sim_pairs['adjacent'] = max(E, E.T).tocsr().fill(1)  # spones(max(E, E'))
    sim_pairs['nbr_radii'] = E.max(axis=1).todense()
    return sim_pairs


def compute_dis(tree: dict, sim_pairs: dict) -> dict:
    """

    Args:
        tree:
        sim_pairs:

    Returns:

    """
    # recurse
    (row_d, col_d, w_d, delta_sq_d) = compute_dis_nbr_radii(tree, sim_pairs['nbr_radii'])

    # do not repel adjacent nodes
    indx = np.ravel_multi_index(row_d, col_d)  # TODO: check this, it is most likely wrong
    repel = np.where(sim_pairs['adjacent'][indx] == 0)[0]
    row_d = row_d[repel]
    col_d = col_d[repel]
    w_d = w_d[repel]
    delta_sq_d = delta_sq_d[repel]

    # normalize
    w_d = w_d / sum(w_d)

    # more efficient for fast sparse matrix construction
    hstacked = np.hstack((col_d, row_d))
    ii = np.lexsort(np.fliplr(hstacked).T)

    # dict
    dis_pairs = {}
    dis_pairs['rows'] = row_d[ii]
    dis_pairs['cols'] = col_d[ii]
    dis_pairs['weights'] = w_d[ii]
    dis_pairs['delta_sq'] = delta_sq_d[ii]
    return dis_pairs



