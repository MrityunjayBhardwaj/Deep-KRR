import torch
# TODO: make a seperate file for kernel definations
def polyKernel(x,y, dim, isMatValued = False):
  #hyp param
  degree = 2
  diag_a  = torch.diag(torch.ones(dim))

  polyKerEval = (torch.matmul(x, y.T) + 1)**degree

  if isMatValued and (dim > 1):
    return diag_a*polyKerEval
  return polyKerEval
