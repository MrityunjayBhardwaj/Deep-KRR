from helper_fns import compose, fn_init
import torch
import torch.nn as nn
from base_kernels import polyKernel
class CompositeKernelRegression(nn.Module):
    def __init__(self, domains, ranges, inputs):
        """
        specify the domain and range of all RKHS + inputs to index the subspaces @ all layers(RKHS).
        """
        super(CompositeKernelRegression, self).__init__()
        # specify kernels for each layers.
        self.K2 = polyKernel
        self.K1 = polyKernel
        self.N = len(inputs)

        # specify the weights for each layers.
        self.K1_weights = torch.randn([self.N, 1, ranges[1]], requires_grad=True) 
        self.K2_weights = torch.randn([self.N, 1, ranges[0]], requires_grad=True) 

        # specify the function at each kernel layer.
        self.fn2 = fn_init(self.K2, self.K2_weights, ranges[0])
        self.fn1 = fn_init(self.K1, self.K1_weights, ranges[1])

        # final composite function.
        self.h = compose([self.fn1, self.fn2], inputs)

    def forward(self, X):
        h_at_all_X = self.h(X[0])
        N = X.shape[0]
        for j in range(1, N):
          h_at_j = self.h(X[j]) # realization of composite function @ x[j]
          h_at_all_X = torch.cat([h_at_all_X, h_at_j ])
        return h_at_all_X

    def parameters(self):
        return [self.K1_weights, self.K2_weights]

