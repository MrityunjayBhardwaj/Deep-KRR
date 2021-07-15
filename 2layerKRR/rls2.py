from compositeKRR import CompositeKernelRegression, SingleLayerKRR
import math
import torch
import torch.nn as nn
import torchviz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gpytorch as gpy

def train_loop(data_x, data_y, model, loss_fn, optimizer, num_epochs=100):
    for epoch in range(num_epochs):

      # Backpropagation
      optimizer.zero_grad()
      # Compute prediction and loss
      pred = model(data_x)
      loss = loss_fn(pred, data_y)
      loss.backward()
      optimizer.step()

      if(loss < 1):
          print('converged! ', loss)
          return True

      if epoch % (1000 + 0*num_epochs) == 0:
        loss = loss.item()
        print(f"{ epoch } loss: {loss:>7f}]")



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

    return [domains, ranges, x_grid, output_1]
def train_4gpy(train_x, train_y, model, likelihood,device, num_epochs ):

    training_iter = num_epochs
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

# Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)

    train_y = train_y.squeeze(1)
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()

        if i % 1000:
          print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
              i + 1, training_iter, loss.item(),
              model.covar_module.base_kernel.lengthscale.item(),
              model.likelihood.noise.item()
          ))
        optimizer.step()
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
    #train_loop(data_x, data_y, model, loss, optimizer, num_epochs)
    train_4gpy(data_x, data_y, model, likelihood, device, num_epochs)

    predY = model(data_x)
    print('predY from SKRR: ', predY)

    return model

def repr_fig3(num_data_points=5,num_epochs=10000):
    """
    Reproducing the result shown in figure3
    """

    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print('Using {} device'.format(device))


    # specify the data.
    _,ranges,data_x,data_y = createSyntheticData(num_data_points)

    data_x = data_x.to(device)
    data_y = data_y.to(device)
    model = e2eKRR(data_x, data_y, ranges, device, num_epochs*0 + 1000);
    model_s = e2eSKRR(data_x, data_y, device, num_epochs*0 + 100);

    # Calculate the predictions.
    pred_y = model(data_x)
    pred_y_s = model_s(data_x).mean

    # calculating the loss at each point.
    loss = torch.abs(pred_y - data_y)
    loss_s = torch.abs(pred_y_s - data_y.squeeze(1)).unsqueeze(1)

    # preparing data for visualization.
    l = loss.reshape([num_data_points, num_data_points]).cpu().detach().numpy()
    l2 = loss_s.reshape([num_data_points, num_data_points]).cpu().detach().numpy()
    data_x_grid = data_x.reshape([ int(data_x.shape[0]**(1/2)), int(data_x.shape[0]**(1/2)), data_x.shape[1]])

    dx1 = dx2 = data_x_grid[:, :, 1][:1, :].squeeze().cpu().detach().numpy()

    print(dx1, dx2, l.shape)

    fig = go.Figure(data=go.Contour(z=l,x=dx1,y=dx2))

    fig = make_subplots(rows=1, cols=2, subplot_titles=['', ''])
    fig.add_trace(go.Contour(z=l2,x=dx1,y=dx2))
    fig.add_trace(go.Contour(z=l,x=dx1,y=dx2))


    fig.show()

    del data_x
    del data_y
    del model
    del dx1

def vizLossLandscape(loss, x, y, z):
    """
    dim(data_x) = 2
    """
    fig = make_subplots(rows=1, cols=2, specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['loss 1', 'loss 2'],)
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorbar_x=-0.07), 1, 1)

    fig.show()


def main(num_epochs=100):
    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    _,ranges,data_x,data_y = createSyntheticData()
    return e2eKRR( data_x, data_y, ranges, device, num_epochs)

def main2(num_epochs=100):
    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print('Using {} device'.format(device))
    _,ranges,data_x,data_y = createSyntheticData()

    # Training data is 100 points in [0,1] inclusive regularly spaced
    #data_x = torch.linspace(0, 1, 100)

    # True function is sin(2*pi*x) with Gaussian noise
    #data_y = torch.sin(data_x * (2 * math.pi)) + torch.randn(data_x.size()) * math.sqrt(0.04)

    print('data shape: ', data_x.shape, data_y.shape)
    return e2eSKRR(data_x,data_y,device,num_epochs)

# main2()
def e2eKRR( data_x, data_y, ranges, device, num_epochs=100):

    # hyperparams
    learning_rate = 0.0005
    #num_epochs = 100000


    # data_to_device
    data_x = data_x.to(device)
    data_y = data_y.to(device)

    print('data_x.device: ', data_x.device, device)

    # initializing the main training loop components.
    model = CompositeKernelRegression(ranges, data_x, device) # TODO: specify the args
    model = model.to(device)

    #predY = model(data_x)

    #print(predY.sum().shape, model.named_parameters())

    print(model.parameters()[0].is_leaf, model.parameters()[1].is_leaf)

    #p = torchviz.make_dot(predY.sum(), params=dict({"K1_weights":model.parameters()[0], 
    #                                                "K2_weights": model.parameters()[1] }),
    #                      show_attrs=True)

    #p.render('somefile.gv', view=True)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_loop(data_x, data_y, model, loss, optimizer, num_epochs)

    #predY = model(data_x)

    #print(predY, data_y)

    #del data_x
    #del data_y
    #del model
    torch.cuda.empty_cache()
    return model

#repr_fig3()
#main(nEpochs)

repr_fig3()
    