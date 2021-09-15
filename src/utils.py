from numpy.core.numeric import ones
from compositeKRR import SingleLayerKRR, DeepKernelRegression
import math
import torch
import torch.nn as nn
import torchviz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gpytorch as gpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def remap( x, oMin, oMax, nMin, nMax ):
	
    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if nMin == nMax:
        print("Warning: Zero output range")
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False   
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result

def quad_as_rect(quad):
    if quad[0] != quad[2]: return False
    if quad[1] != quad[7]: return False
    if quad[4] != quad[6]: return False
    if quad[3] != quad[5]: return False
    return True

def quad_to_rect(quad):
    assert(len(quad) == 8)
    assert(quad_as_rect(quad))
    return (quad[0], quad[1], quad[4], quad[3])

def rect_to_quad(rect):
    assert(len(rect) == 4)
    return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])

def shape_to_rect(shape):
    assert(len(shape) == 2)
    return (0, 0, shape[0], shape[1])

def griddify(rect, w_div, h_div):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    x_step = w / float(w_div)
    y_step = h / float(h_div)
    y = rect[1]
    grid_vertex_matrix = []
    for _ in range(h_div + 1):
        grid_vertex_matrix.append([])
        x = rect[0]
        for _ in range(w_div + 1):
            grid_vertex_matrix[-1].append([int(x), int(y)])
            x += x_step
        y += y_step
    grid = np.array(grid_vertex_matrix)
    return grid

def distort_grid(org_grid, max_shift):
    new_grid = np.copy(org_grid)
    x_min = np.min(new_grid[:, :, 0])
    y_min = np.min(new_grid[:, :, 1])
    x_max = np.max(new_grid[:, :, 0])
    y_max = np.max(new_grid[:, :, 1])
    new_grid += np.random.randint(- max_shift, max_shift + 1, new_grid.shape)
    new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
    new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
    return new_grid

def grid_to_mesh(src_grid, dst_grid):
    assert(src_grid.shape == dst_grid.shape)
    mesh = []
    for i in range(src_grid.shape[0] - 1):
        for j in range(src_grid.shape[1] - 1):
            src_quad = [src_grid[i    , j    , 0], src_grid[i    , j    , 1],
                        src_grid[i + 1, j    , 0], src_grid[i + 1, j    , 1],
                        src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
                        src_grid[i    , j + 1, 0], src_grid[i    , j + 1, 1]]

            dst_quad = [dst_grid[i    , j    , 0], dst_grid[i    , j    , 1],
                        dst_grid[i + 1, j    , 0], dst_grid[i + 1, j    , 1],
                        dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
                        dst_grid[i    , j + 1, 0], dst_grid[i    , j + 1, 1]]
            dst_rect = quad_to_rect(dst_quad)
            mesh.append([dst_rect, src_quad])
    return mesh

def train_loop(data_x, data_y, model, loss_fn, optimizer, num_epochs=100, is_neg_loss=0, scheduler=None):
    threshold = 1

    print('training_epochs: ', num_epochs)
    prev_loss = 0
    for epoch in range(num_epochs):

      # Backpropagation
      optimizer.zero_grad()
      # Compute prediction and loss
      pred = model(data_x)
      loss = loss_fn(pred, data_y)

      if is_neg_loss:
          loss = -1*loss
      is_last_epoch = (num_epochs - 1 ) == epoch or (loss < threshold)

      if torch.abs(prev_loss - loss) < 0.00000001:
          pass
          print('converged!')
          #break
      prev_loss = loss

      loss.backward()
      optimizer.step()

      if scheduler:
          print('scheduler step')
          scheduler.step()


      if epoch % (1000 + 0*num_epochs) == 0:
        loss = loss.item()
        print(f"{ epoch } loss: {loss:>7f}]")

      if(is_last_epoch):
          return True


def createSyntheticData(num_data_points=10):
    #following the defination specified in the paper.
    input_domain = {'x_0': [-1, 1],
             'x_1': [-1, 1]}

    def test_fn_1(x_0, x_1):
      return (0.1 + torch.abs(x_0 - x_1))**(-1)

    def test_fn_2(x_0, x_1):
      return 1 if (x_0*x_1 > 3/20) else 0

    # initializing the input domains
    x_0 = torch.linspace(input_domain['x_0'][0], input_domain['x_0'][1],num_data_points)
    x_1 = torch.linspace(input_domain['x_1'][0], input_domain['x_1'][1],num_data_points)

    #initializing the grid and outputs
    x_grid= []
    output_1 = []
    output_2 = []


    for x in x_0:
      out_1_row = []
      out_2_row = []
      for y in x_1:
        # calculating the ouptuts
        out_1 = test_fn_1(x, y)
        out_2 = test_fn_2(x, y)
        # appending the input values and fn outputs.
        x_grid.append([x,y])
        out_1_row.append(out_1)
        out_2_row.append(out_2)
      output_1.append(out_1_row)
      output_2.append(out_2_row)

    # converting the inputs and outputs to torch tensor with appropriate dimensions.
    x_grid = torch.tensor(x_grid).reshape(num_data_points**2, 2)
    output_1 = torch.tensor(output_1).reshape(num_data_points**2, 1)
    output_2 = torch.tensor(output_2).reshape(num_data_points**2, 1)

    domains = [2, 2]
    ranges  = [2, 1]

    return [domains, ranges, x_grid, output_1, output_2]

def e2eSKRR(data_x, data_y, device, num_epochs=100):
    # TODO: create single layer KRR

    learning_rate = 0.0005

    # data_to_device
    data_x = data_x.to(device)
    data_y = data_y.to(device)

    likelihood = gpy.likelihoods.GaussianLikelihood()
    model = SingleLayerKRR(data_x,data_y, likelihood).to(device)

    print('SKRR params: ', model.parameters())

    loss = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    decayRate = 0.96
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    data_y = data_y.squeeze(1)
    train_loop(data_x, data_y, model, loss, optimizer,num_epochs, is_neg_loss=1,  scheduler=lr_scheduler )

    predY = model(data_x)
    print('predY from SKRR: ', predY)

    return model


def test_e2eKRR(num_epochs=100):
    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    _,ranges,data_x,data_y = createSyntheticData()
    return e2eKRR( data_x, data_y, ranges, device, num_epochs)

def init_rls2_model(ranges, data_x, degree, device, retain_layer_outputs=False):
    """
    define and initialize the model for experiment defined in section 4.2
    """
    # specify kernels for each layers.
    K0 = gpy.kernels.PolynomialKernel(degree) # inner Kernel
    K1 = gpy.kernels.MaternKernel() # outer kernel
    kernels = [K0, K1]

    model = DeepKernelRegression(ranges, data_x, kernels, device, retain_layer_outputs=retain_layer_outputs)
    model = model.to(device)

    return model

def e2eKRR( data_x, data_y, ranges, degree, device, num_epochs=100, model_path=None, load_model=False,save_model=False, retain_layer_outputs=False, vizCompGraph=False):

    # hyperparams
    learning_rate = 0.0005

    # data_to_device
    data_x = data_x.to(device)
    data_y = data_y.to(device)

    print('e2eKRR num_epochs: ', num_epochs)
    print('data_x.device: ', data_x.device, device)

    # initializing the main training loop components.
    model = init_rls2_model(ranges, data_x, degree, device, retain_layer_outputs=retain_layer_outputs)
    predY = model(data_x)
    print(model.parameters()[0].is_leaf, model.parameters()[1].is_leaf)

    if vizCompGraph:
        p = torchviz.make_dot(predY.sum(), params=dict({"K1_weights":model.parameters()[0], 
                                                        "K2_weights": model.parameters()[1] }),
                              show_attrs=True, show_saved=True)

        p.render('somefile'+str(degree)+'.gv', view=True)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if load_model:
        checkpoint = torch.load(model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        num_epochs = checkpoint['epoch']
        loss = checkpoint['loss']
        params = checkpoint['params']
        model.load_params(params)
        print('model sucessfully loaded! from '+model_path)

    train_loop(data_x, data_y, model, loss, optimizer, num_epochs,is_neg_loss=0)

    if save_model:
        torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'params' : model.parameters(),
                }, model_path)
        print('model saved! @ '+model_path)

    torch.cuda.empty_cache()
    return model
