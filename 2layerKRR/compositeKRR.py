from vectorized import compose, fn_init
import torch
import torch.nn as nn
import gpytorch as gpy
import torchviz

Vectorized all the calculation of compositeKRR

* vectorized main calculation, removing all the loops and serialized
  calcuation.
* Assuming kernel fn is coming from gpytorch library.
* removing the preprocessing step in compose function to fix the
  computation graph and unnecessary dual calculation.
* no calculation takes place before calling return function of the
    compose fn.

class CompositeKernelRegression(nn.Module):
    def __init__(self, ranges, inputs, device="cpu"):
        """
        specify the domain and range of all RKHS + inputs to index the subspaces @ all layers(RKHS).
        """
        super(CompositeKernelRegression, self).__init__()
        # specify kernels for each layers.
        self.K2 = gpy.kernels.PolynomialKernel(2).to(device)
        self.K1 = gpy.kernels.PolynomialKernel(2).to(device)

        self.N  = len(inputs)

        # specify the weights for each layers.
        self.K1_weights = torch.randn([self.N, 1, ranges[1]], requires_grad=True, device=device)
        self.K2_weights = torch.randn([self.N, 1, ranges[0]], requires_grad=True, device=device)


        # specify the function at each kernel layer.
        self.fn2 = fn_init(self.K2, self.K2_weights, ranges[0], device)
        self.fn1 = fn_init(self.K1, self.K1_weights, ranges[1], device)

        # final composite function.
        self.h = compose([self.fn1, self.fn2], inputs)

    def forward(self, X):
        return self.h(X)

    def parameters(self):
        return [self.K1_weights, self.K2_weights]

