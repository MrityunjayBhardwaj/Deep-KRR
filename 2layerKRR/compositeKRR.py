from vectorized import compose, fn_init
import torch
import torch.nn as nn
import gpytorch as gpy
import torchviz

class SingleLayerKRR(gpy.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SingleLayerKRR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpy.means.ConstantMean()
        #self.covar_module = gpy.lazy.ZeroLazyTensor(train_x.size())
        #self.covar_module  = gpy.kernels.PolynomialKernel(2)
        self.covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)

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

