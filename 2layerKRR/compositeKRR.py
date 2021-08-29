from vectorized import compose, fn_init
import torch
import torch.nn as nn
import gpytorch as gpy
import torchviz

class SingleLayerKRR(gpy.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SingleLayerKRR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpy.means.ConstantMean()
        self.covar_module  = gpy.kernels.PolynomialKernel(2)

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)

class CompositeKernelRegression(nn.Module):
    def __init__(self, ranges, inputs, device="cpu",**kwargs):
        """
        specify the domain and range of all RKHS + inputs to index the subspaces @ first RKHS
        """
        super(CompositeKernelRegression, self).__init__()

        poly_degree_K2 = kwargs.get('degree') or 2
        retain_layer_outputs = kwargs.get('retain_layer_outputs')

        # specify kernels for each layers.
        self.K2 = gpy.kernels.PolynomialKernel(poly_degree_K2).to(device)
        #self.K1 = gpy.kernels.PolynomialKernel(poly_degree_K1).to(device)
        self.K1 = gpy.kernels.MaternKernel().to(device)

        self.N  = len(inputs)

        # specify the weights for each layers.
        self.K1_weights = torch.randn([self.N, 1, ranges[1]], requires_grad=True, device=device)
        self.K2_weights = torch.randn([self.N, 1, ranges[0]], requires_grad=True, device=device)

        print('kernel', self.K2, self.K1, ranges)

        # specify the function at each kernel layer.
        self.fn1 = fn_init(self.K1, self.K1_weights, ranges[1], device)
        self.fn2 = fn_init(self.K2, self.K2_weights, ranges[0], device)

        # final composite function.
        self.h = compose([self.fn2, self.fn1], inputs, retain_layer_outputs)

        self.layer_outputs = []

    def forward(self, X):
        final_output, all_layers_outputs = self.h(X)

        self.layer_outputs = all_layers_outputs
        return final_output

    def parameters(self):
        return [self.K1_weights, self.K2_weights]


class DeepKernelRegression(nn.Module):
    def __init__(self, ranges, inputs, kernels, device="cpu",**kwargs):
        """
        specify the domain and range of all RKHS + inputs to index the subspaces @ first RKHS.

        kwargs:
            retain_layer_outputs: store the outputs of each kernel layer.
        """
        super(DeepKernelRegression, self).__init__()

        retain_layer_outputs = kwargs.get('retain_layer_outputs')
        self.Kernels = kernels
        self.Weights = []
        self.N  = len(inputs)
        self.Fns = []

        for i, kernel in enumerate(self.Kernels):
            print(i, 'kernel', kernel, ranges[i])
            self.Kernels[i] = kernel.to(device)
            # specify weights for each layers
            curr_Ker_weights =  torch.randn([self.N, 1, ranges[i]], requires_grad=True, device=device)
            self.Weights.append(curr_Ker_weights)
            # specify the function at each kernel layer.
            curr_Ker_Fn =  fn_init(self.Kernels[i], self.Weights[i], ranges[i], device)
            self.Fns.append(curr_Ker_Fn)

        self.compositeFn = compose(self.Fns, inputs, retain_layer_outputs)

        self.layer_outputs = []

    def forward(self, X):
        final_output, all_layers_outputs = self.compositeFn(X)

        self.layer_outputs = all_layers_outputs
        return final_output

    def parameters(self):
        return self.Weights

class RLS2(DeepKernelRegression):
    def __init__(self, ranges, inputs, device="cpu", **kwargs):
        """
        2 layer composite Kernels to reproduce the results in _ et al. (2018)
        """

        poly_degree_K2 = kwargs.get('degree') or 2
        retain_layer_outputs = kwargs.get('retain_layer_outputs') or None

        # specify kernels for each layers.
        K2 = gpy.kernels.PolynomialKernel(poly_degree_K2).to(device)
        #self.K1 = gpy.kernels.PolynomialKernel(poly_degree_K1).to(device)
        K1 = gpy.kernels.MaternKernel().to(device)
        kernels = [K2, K1]
        super().__init__(ranges, inputs, kernels, device=device, retain_layer_outputs=retain_layer_outputs)
