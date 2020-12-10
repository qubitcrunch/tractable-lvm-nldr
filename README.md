# tractable-lvm-nldr
An attempt at Python implementation of A Tractable Latent Variable model for Nonlinear Dimensionality Reduction - Lawrence
                K. Saul. 
Pre-print available here - https://cseweb.ucsd.edu/~saul/pnas20/preprint.pdf

# Project Structure
dist_helpers.py -> Houses some helpers that calculate a variety of distance function. The choice of distance function is
                    ultimately reflected in knn_and_graph_funcs.compute_dis           
                    
em_helpers.py -> Houses the init_graph function that returns an initial value of \mu0 as well as the EM-learning 
                    algorithm
                    
knn_and_graph_funcs.py -> Houses various KNN functions, functions to calculate similarity and dis-similarity pairs 
                            given a tree
                            
main.py -> the main file that runs the algorithm step-by-step