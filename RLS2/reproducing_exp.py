import torch
import numpy as np

def rand_diag_matrix(dim):
  """
  returns a random diagonalized matrix 
  """
  vec = torch.randn(dim)
  return torch.diag(vec)
    

def polyKernel(x,y, dim, isMatValued = False):
  #hyp param
  degree = 2
  diag_a  = torch.diag(torch.ones(dim))

  polyKerEval = (torch.matmul(x, y.T) + 1)**degree

  if isMatValued and (dim > 1):
    return diag_a*polyKerEval
  return polyKerEval

def gaussianKernel(x,y, isMatValued = False):
  dim = x.shape[0]
  diag_a  = torch.ones(dim)

  sigma = 0.1
  gaussianKerEval = torch.exp(-(torch.norm(x-y)/(2*sigma**2)))

  if isMatValued:
    return diag_a*gaussianKerEval
  return gaussianKerEval

# TODO: fix this. the kappa is the modified bessel function of second kind.
def tensorMatern(x,y):
  d = x.shape[0]
  print(d)
  
  # hyp params
  kappa = 10
  s = 2



  out = 1
  for i in range(d):
    out *= kappa*((2*s - 1)/2)*( torch.abs(x[i] - y[i]) * torch.abs(x[i] -y[i])**((2*s -1)/2) )
  return out

"""
Defining our compositional Kernels

"""

K2 = polyKernel
K1 = polyKernel
N = x_grid.shape[0]

x_grid = x_grid.reshape([100, 1, 2])

K2_weights = torch.randn([100, 1, range2],requires_grad=True)

def fn2(inputs, val):
  fn2_at_val = 0
  N = len(inputs)
  for i in range(N):
    x_i = inputs[i]
    fn2_at_val += K2_weights[i].matmul(K2(x_i, val, range2, True).T)
  return fn2_at_val

K1_weights = torch.randn([100, 1, range1], requires_grad=True)
def fn1(inputs, val):
  fn1_at_j = 0 # realization of the inner function @ x[j]
  N = len(inputs)
  for i in range(N):
    x_i = inputs[i]
    K1_xi_val = K1(x_i, val, range1, False)
    fn1_at_j += K1_weights[i].matmul(K1_xi_val.T)
  return fn1_at_j


"""
defining the subspace spanned by the basis fns indexed by args.
"""



def composite_fns(ker_arr, dom_arr):
  """
  given an array of kernel functions and domain of each of the rkhs this
  function returns the composite function.
  """
  pass

def lazy_fn(fn, inputs):
  """
  this fn initialize the inputs args of given fn.
  """
  def fn_at_inputs(val):
    return fn(inputs, val)
  return fn_at_inputs

def calc_realization_at_inputs(fn, inputs):
  fn_at_all_inputs = []
  N = inputs.shape[0]
  for j in range(N):
    fn_at_j = fn(inputs[j]) # 'realization' of the function at inputs[j]
    fn_at_all_inputs.append(fn_at_j.numpy())
  return torch.tensor(fn_at_all_inputs)

def preprocess(skeleton_fn_arr, inputs):
  fn_arr = []
  prev_inp = inputs
  for fn in skeleton_fn_arr:
    fn_at_prev_inp = lazy_fn(fn, prev_inp) # calculate the function indexed at all prev_inps.

    # calculate the 'realization' at all prev_inps.
    # if (!is_last_fn)
    # print()
    prev_inp = calc_realization_at_inputs(fn_at_prev_inp, prev_inp)
    # c2 = fn_at_prev_inp

    # append the result
    fn_arr.append(fn_at_prev_inp)
  return fn_arr

def realize_composite_fns(fn_arr, val):
  prev_val = val
  # TODO: reverse the order of fn_arr
  for curr_fn in fn_arr:
    
    curr_val = curr_fn(prev_val)
    # print(curr_fn, prev_val, curr_val)
    prev_val = curr_val
    
  return prev_val

def compose(skel_fn_arr, input):
  fn_arr = preprocess(skel_fn_arr, input) # function in the subspace of rkhs spanned by the repe_eval_at_prev_inputs. NOTE: skeleton of rep eval is present in each skel_fn_arr entry.
  def fn(val):
    return  realize_composite_fns(fn_arr, val)
  return fn
