import torch
from utils import e2eKRR, e2eSKRR, createSyntheticData
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def repr_fig3(num_data_points=5,num_epochs=10000, viz_prediction_only=False):
    """
    Reproducing the result shown in figure3

    Visualizing the loss or only predictions from 2 deep kernel architectures.
    """

    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print('Using {} device'.format(device))


    # specify the data.
    _,ranges,data_x,data_y_h1, data_y_h2 = createSyntheticData(num_data_points)

    data_x = data_x.to(device)
    data_y_h1 = data_y_h1.to(device)
    data_y_h2 = data_y_h2.to(device)*1.0

    # num_epochs = 1000

    print(data_x.shape, data_y_h1.shape, data_y_h2.shape)


    # calculating the models for constructing h1 and h2 functions.
    first_layer_poly_kernel_degree = 1
    model_comp_h1_v1 = e2eKRR(data_x, data_y_h1, ranges, first_layer_poly_kernel_degree, device, num_epochs)
    model_comp_h2_v1 = e2eKRR(data_x, data_y_h2*1.0, ranges, first_layer_poly_kernel_degree, device, num_epochs)

    first_layer_poly_kernel_degree = 2
    model_comp_h1_v2 = e2eKRR(data_x, data_y_h1, ranges, first_layer_poly_kernel_degree, device, num_epochs)
    model_comp_h2_v2 = e2eKRR(data_x, data_y_h2, ranges, first_layer_poly_kernel_degree, device, num_epochs)

    model_single_h1 = e2eSKRR(data_x, data_y_h1, device, num_epochs)
    model_single_h2 = e2eSKRR(data_x, data_y_h2, device, num_epochs)

    # Calculate the predictions.
    pred_y_comp_h1_v1 = model_comp_h1_v1(data_x)
    pred_y_comp_h2_v1 = model_comp_h2_v1(data_x)
    pred_y_comp_h1_v2 = model_comp_h1_v2(data_x)
    pred_y_comp_h2_v2 = model_comp_h2_v2(data_x)

    pred_y_single_h1 = model_single_h1(data_x).mean
    pred_y_single_h2 = model_single_h2(data_x).mean

    # calculating the loss at each point.
    loss_comp_h1_v1 = torch.abs(pred_y_comp_h1_v1 - data_y_h1*viz_prediction_only)
    loss_comp_h2_v1 = torch.abs(pred_y_comp_h2_v1 - data_y_h2*viz_prediction_only)
    loss_comp_h1_v2 = torch.abs(pred_y_comp_h1_v2 - data_y_h1*viz_prediction_only)
    loss_comp_h2_v2 = torch.abs(pred_y_comp_h2_v2 - data_y_h2*viz_prediction_only)

    loss_single_h1 = torch.abs(pred_y_single_h1 - data_y_h1.squeeze(1)).unsqueeze(1)
    loss_single_h2 = torch.abs(pred_y_single_h2 - data_y_h2.squeeze(1)).unsqueeze(1)

    # preparing data for visualization.
    loss_comp_h1_v1_viz = loss_comp_h1_v1.reshape([num_data_points, num_data_points]).cpu().detach().numpy()
    loss_comp_h2_v1_viz = loss_comp_h2_v1.reshape([num_data_points, num_data_points]).cpu().detach().numpy()
    loss_comp_h1_v2_viz = loss_comp_h1_v2.reshape([num_data_points, num_data_points]).cpu().detach().numpy()
    loss_comp_h2_v2_viz = loss_comp_h2_v2.reshape([num_data_points, num_data_points]).cpu().detach().numpy()

    loss_single_h1_viz = loss_single_h1.reshape([num_data_points, num_data_points]).cpu().detach().numpy()
    loss_single_h2_viz = loss_single_h2.reshape([num_data_points, num_data_points]).cpu().detach().numpy()

    data_x_grid = data_x.reshape([ int(data_x.shape[0]**(1/2)), int(data_x.shape[0]**(1/2)), data_x.shape[1]])
    dx1 = dx2 = data_x_grid[:, :, 1][:1, :].squeeze().cpu().detach().numpy()

    fig = make_subplots(rows=2, cols=3,row_titles=['Function A', 'Function B'], subplot_titles=['original fn', '(p = 1)°Matern', ' (p = 2)°Matern'])
    fig.add_trace(go.Contour(z=loss_single_h1_viz,x=dx1,y=dx2), 1, 1)
    fig.add_trace(go.Contour(z=loss_single_h2_viz,x=dx1,y=dx2), 2, 1)

    fig.add_trace(go.Contour(z=loss_comp_h1_v1_viz,x=dx1,y=dx2), 1, 2)
    fig.add_trace(go.Contour(z=loss_comp_h2_v1_viz,x=dx1,y=dx2), 2, 2)
    fig.add_trace(go.Contour(z=loss_comp_h1_v2_viz,x=dx1,y=dx2), 1, 3)
    fig.add_trace(go.Contour(z=loss_comp_h2_v2_viz,x=dx1,y=dx2), 2, 3)

    fig.show()
