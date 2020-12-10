import argparse
import numpy as np
from knn_and_graph_funcs import *
from em_helpers import *


def load_data(file_path: str) -> np.ndarray:
    """

    Args:
        file_path:

    Returns:

    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run tractable LVM for Nonlinear-dimensionality reduction')
    parser.add_argument('data-file', type=str, required=True, help='absolute path of the file containing data')
    parser.add_argument('k', type=int, required=True, help='number of nearest neighbors')
    parser.add_argument('s', type=int, required=True, help='steps along nearest neighbor graph')
    parser.add_argument('l', type=int, required=True, help='levels of coarse-graining')
    parser.add_argument('d', type=int, required=True, help='dimensionality of embedding')
    parser.add_argument('max-iter-em', type=int, default=400)
    parser.add_argument('momentum', type=float, default=0.9)
    parser.add_argument('print-cost', type=int, default=10)
    parser.add_argument('cmg-pcg-tol', type=float, default=1e-6)
    parser.add_argument('cmg-pcg-max-iter', type=int, default=50)
    args = parser.parse_args()

    X = load_data(args.data_file)
    E = compute_neighborhood_graph(X, args.k, args.s)
    # coarse graining
    cg_tree = coarse_grain(X, args.l)
    # similar and dissimilar examples
    sim_pairs = compute_sim(cg_tree, E)
    dis_pairs = compute_dis(cg_tree, sim_pairs)
    # initiliaze
    mu0 = init_from_graph_laplacian(sim_pairs, d)
    # learn
    (mu, sigma_sq) = learn(mu0, sim_pairs, dis_pairs, args)


