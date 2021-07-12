from compositeKRR import CompositeKernelRegression
import torch
import torch.nn as nn
import torchviz

def train_loop(data_x, data_y, model, loss_fn, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
      # Compute prediction and loss
      pred = model(data_x)
      loss = loss_fn(pred, data_y)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()

      if(loss < 1):
          print('converged! ', loss)
          return True

      if epoch % (1000 + 0*num_epochs) == 0:
        loss = loss.item()
        print(f"{ epoch } loss: {loss:>7f}]")



def createSyntheticData():
    #following the defination specified in the paper.
    input_domain = {'x_0': [-1, 1],
             'x_1': [-1, 1]}

    def test_fn_1(x_0, x_1):
      return (0.1 + torch.abs(x_0 - x_1))**(-1)

    def test_fn_2(x_0, x_1):
      return 1 if (x_0*x_1 > 3/20) else 0


    num_data_points = 10
        
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
    x_grid = torch.tensor(x_grid).reshape(100, 2)
    output_1 = torch.tensor(output_1).reshape(100, 1)
    output_2 = torch.tensor(output_2).reshape(100, 1)

    domains = [2, 2]
    ranges  = [2, 1]

    return [domains, ranges, x_grid, output_1]


def main():

    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print('Using {} device'.format(device))

    # hyperparams
    learning_rate = 0.0005
    num_epochs = 100000


    # specify the data.
    _,ranges,data_x,data_y = createSyntheticData()
    data_x = data_x.to(device)
    data_y = data_y.to(device)

    print('data_x.device: ', data_x.device, device)

    # initializing the main training loop components.
    model = CompositeKernelRegression(ranges, data_x, device) # TODO: specify the args
    model = model.to(device)

    predY = model(data_x)

    print(predY.sum().shape, model.named_parameters())

    print(model.parameters()[0].is_leaf, model.parameters()[1].is_leaf)

    p = torchviz.make_dot(predY.sum(), params=dict({"K1_weights":model.parameters()[0], 
                                                    "K2_weights": model.parameters()[1] }),
                          show_attrs=True)

    p.render('somefile.gv', view=True)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


    train_loop(data_x, data_y, model, loss, optimizer, num_epochs)

    predY = model(data_x)

    #print(predY, data_y)

    del data_x
    del data_y
    del model
    torch.cuda.empty_cache()

main()
