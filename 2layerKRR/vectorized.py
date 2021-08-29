import torch
import gpytorch

def polyKernel(d):
    # following the implementation, similar to gpy.kernel.PolynomialKernel
    # x1 : 100, 2 and x2: 70, 2
    # out => 100, 7
    def _fn(x1, x2):
        return torch.matmul(x1, x2.T).pow(d)
    return _fn

# TODO: vectorize the Kernel
# TODO: vectorize the fn
def ker(x1, x2, ker, rkhs_range, device):
    k = ker(x1, x2)
    k = k.evaluate() # if using gpytorch
    k = k.unsqueeze(2) 
    k = k.expand(*k.shape[:2], rkhs_range) # 100, 70, 2
    k = k.unsqueeze(2) # 100, 70, 1,2
    ey = torch.eye(rkhs_range).unsqueeze(2).expand(rkhs_range,rkhs_range, k.shape[0]).T
    ey = torch.eye(rkhs_range).reshape(1,1, rkhs_range, rkhs_range).expand(*k.shape[: 2], rkhs_range, rkhs_range).to(device)
    return k.mul(ey)

def rkhs_fn(X,Y,base_kernel, weights, rkhs_range, device):
    k = ker(X, Y, base_kernel, rkhs_range, device)
    return weights.matmul(k).sum(1).squeeze(1) # realization at Y of weight linear combination of basis functions indexed by X

def fn_init(Kernel, Ker_weights, range, device):
    """
    this function simply set the args for our main 'fn' function.
    """

    def fn_with_specified_args(inputs, val):
        return rkhs_fn(inputs, val, Kernel, Ker_weights, range, device)
    return fn_with_specified_args

def lazy_fn(fn, inputs):
  """
  this fn initialize the inputs args of given fn.
  """
  def fn_at_inputs(val):
    return fn(inputs, val)
  return fn_at_inputs

def calc_realization_at_inputs(fn, inputs):
  fn_at_all_inputs = fn(inputs)
  return fn_at_all_inputs

def preprocess(skeleton_fn_arr, inputs):
  fn_arr = []
  prev_inp = inputs

  for fn in skeleton_fn_arr:
    fn_at_prev_inp = lazy_fn(fn, prev_inp) # calculate the function indexed at all prev_inps.

    # calculate the 'realization' at all prev_inps.
    prev_inp = calc_realization_at_inputs(fn_at_prev_inp, prev_inp)

    # append the result
    fn_arr.append(fn_at_prev_inp)
  return fn_arr

def realize_composite_fns(fn_arr, input, val, retain_layer_outputs=False):
  prev_val = val

  layer_outputs = []
  # TODO: reverse the order of fn_arr
  for curr_fn in fn_arr:
    
    curr_val = curr_fn(input, prev_val)
    # print(curr_fn, prev_val, curr_val)
    prev_val = curr_val

    if(retain_layer_outputs):
        layer_outputs.append(prev_val)

  return prev_val,layer_outputs

def compose(skel_fn_arr, input, retain_layer_outputs):
  """
  given a array of fns this function calculates the composition of these function in mathematically consistent fashion
  i.e, compose([fn1, fn2, fn3], x) == fn1 ° fn2 ° fn3 == fn1(fn2(fn3(x))) respectively.
  """
  fn_arr = skel_fn_arr

  if (retain_layer_outputs):
      print('retaining layer outputs')
  #fn_arr = preprocess(skel_fn_arr, input) # function in the subspace of rkhs spanned by the repe_eval_at_prev_inputs. NOTE: skeleton of rep eval is present in each skel_fn_arr entry.
  def fn(val):
    return  realize_composite_fns(fn_arr,input, val, retain_layer_outputs)
  return fn
