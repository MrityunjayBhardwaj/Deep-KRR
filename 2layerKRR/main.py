import torch
import numpy as np

num_data_points = 100
X = torch.ones(num_data_points, 1)
X = 

def concat_kernel_learning(X, Y, K2, K1, Loss):
    """
    Args:
        X_i: 1xd
        K2: X_i x X_i => z_i : 1xp
        K1: z_i x z_i => t_i : 1x1
    TODO: 
        - get the inputs, Kernels
        - take the linear combination of K2
        - put the answer in K1
        - output the predicted y
    """
    input_shape = X.shape
    output_shape = Y.shape

    # initializing weights
    W_0 = torch.randn()

    # Since 2 arg must be disjoint we need to create 2 seperate version of it.
    X_1, X_2 = X, X
    Z = torch.dot(K2(X_1, X_2), W_0)

    Z_1, Z_2 = Z, Z
    Y_hat = K2(Z_1,Z_2)

    cLoss = Loss(Y_hat, Y)

    return cLoss
