import torch
def fn_init(Kernel, Ker_weights, range):
    """
    this function simply set the args for our main 'fn' function.
    """

    def fn_with_specified_args(inputs, val):
        return fn(inputs, val, Kernel, Ker_weights, range)
    return fn_with_specified_args

def fn(inputs, val, Kernel, Ker_weights, rkhs_range):
  fn_at_val = 0
  N = len(inputs)

  for i in range(N):
    x_i = inputs[i]
    fn_at_val += Ker_weights[i].matmul(Kernel(x_i, val, rkhs_range, True).T)
  return fn_at_val

def lazy_fn(fn, inputs):
  """
  this fn initialize the inputs args of given fn.
  """
  def fn_at_inputs(val):
    return fn(inputs, val)
  return fn_at_inputs

def calc_realization_at_inputs(fn, inputs):
  fn_at_all_inputs = fn(inputs[0])
  N = inputs.shape[0]
  for j in range(1, N):
    
    fn_at_j = fn(inputs[j]) # 'realization' of the function at inputs[j]
    fn_at_all_inputs = torch.cat([fn_at_all_inputs, fn_at_j])
  fn_at_all_inputs = fn_at_all_inputs.reshape([fn_at_all_inputs.shape[0], 1, fn_at_all_inputs.shape[1]])
  print(fn_at_all_inputs.shape)
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

def realize_composite_fns(fn_arr, val):
  prev_val = val
  # TODO: reverse the order of fn_arr
  for curr_fn in fn_arr:
    
    curr_val = curr_fn(prev_val)
    # print(curr_fn, prev_val, curr_val)
    prev_val = curr_val
    
  return prev_val

def compose(skel_fn_arr, input):
  """
  given a array of fns this function calculates the composition of these function in mathematically consistent fashion
  i.e, compose([fn1, fn2, fn3], x) == fn1 ° fn2 ° fn3 == fn1(fn2(fn3(x))) respectively.
  """
  skel_fn_arr.reverse() # reversing the order to make it consistent with the mathematical notation.
  fn_arr = preprocess(skel_fn_arr, input) # function in the subspace of rkhs spanned by the repe_eval_at_prev_inputs. NOTE: skeleton of rep eval is present in each skel_fn_arr entry.
  def fn(val):
    
    return  realize_composite_fns(fn_arr, val)
  return fn
