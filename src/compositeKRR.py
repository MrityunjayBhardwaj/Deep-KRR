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

        # private variables
        self._ranges = ranges
        self._device = device
        self._inputs = inputs
        self._retain_layer_outputs = retain_layer_outputs

        for i, kernel in enumerate(self.Kernels):
            print(i, 'kernel', kernel, ranges[i])
            self.Kernels[i] = kernel.to(device)
            # specify weights for each layers
            curr_Ker_weights =  torch.randn([self.N, 1, ranges[i]], requires_grad=True, device=device)
            self.Weights.append(curr_Ker_weights)

        self.layer_outputs = []

    def forward(self, X):
        self.Fns = []
        for i, kernel in enumerate(self.Kernels):
            # specify the function at each kernel layer.
            curr_Ker_Fn =  fn_init(self.Kernels[i], self.Weights[i], self._ranges[i], self._device)
            self.Fns.append(curr_Ker_Fn)
            pass
        self.compositeFn = compose(self.Fns, self._inputs, self._retain_layer_outputs)
        final_output, all_layers_outputs = self.compositeFn(X)

        self.layer_outputs = all_layers_outputs
        return final_output

    def parameters(self):
        return self.Weights

    def load_params(self, new_weights):
        self.Weights = new_weights

