import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from wand.image import Image, Color
from itertools import chain
from utils import e2eKRR, createSyntheticData, remap, griddify, shape_to_rect, grid_to_mesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def repr_fig6_plotly(num_data_points=10, num_epochs=2000):
    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print('Using {} device'.format(device))

    _,ranges,data_x,data_y_h1, data_y_h2 = createSyntheticData(num_data_points)
    data_x = data_x.to(device)
    data_y_h1 = data_y_h1.to(device)
    data_y_h2 = data_y_h2.to(device)*1.0
    data_y_h1 = data_y_h1.to(device)*1.0

    # calculating the models for constructing h1 and h2 functions.
    first_layer_poly_kernel_degree = 2
    model_comp_h1_v1 = e2eKRR(data_x, data_y_h1, ranges, first_layer_poly_kernel_degree, device, num_epochs, model_path='model_ckpt.pth', save_model=True, retain_layer_outputs=True)
    layer_outputs = model_comp_h1_v1.layer_outputs

    l0_out = layer_outputs[0].detach() # layer 0 outputs
    l0_out = l0_out.cpu().detach().numpy()

    x = data_x[:, 0].cpu().detach().numpy()
    y = data_x[:, 1].cpu().detach().numpy()
    z = data_y_h1[:,0].cpu().detach().numpy()
    intensity = remap(z, np.min(z), np.max(z), 0.0, 1.0)

    l0_x = l0_out[:, 0]
    l0_y = l0_out[:, 1]


    fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}],
           ])

    mesh_plot_00 = go.Mesh3d(x=x, y=y, z=z,
                    alphahull=5,
                    opacity=1.0,
                    color='cyan',
                    colorscale=[[0.0, 'blue'],
                                [0.5, 'magenta'],
                                [1.0, 'green']],
                    intensity = intensity
                    )



    mesh_plot_01= go.Mesh3d(x=l0_x, y=l0_y, z=z,
                    alphahull=5,
                    opacity=1.0,
                    color='cyan',
                    colorscale=[[0.0, 'blue'],
                                [0.5, 'magenta'],
                                [1.0, 'green']],
                    intensity = intensity
                    )

    fig.add_trace(mesh_plot_00, row=1, col=1)
    fig.add_trace(mesh_plot_01, row=1, col=2)

    fig.update_layout(
        title_text='Representation learning in deep kernel ridge regression',
        height=800,
        width=800*2
    )
    fig.show()


def repr_fig6_exp(num_data_points=10, num_epochs=10000):
    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print('Using {} device'.format(device))

    _,ranges,data_x,data_y_h1, data_y_h2 = createSyntheticData(num_data_points)
    data_x = data_x.to(device)
    data_y_h1 = data_y_h1.to(device)
    data_y_h2 = data_y_h2.to(device)*1.0

    # calculating the models for constructing h1 and h2 functions.
    first_layer_poly_kernel_degree = 2
    model_comp_h1_v1 = e2eKRR(data_x, data_y_h1, ranges, first_layer_poly_kernel_degree, device, num_epochs, retain_layer_outputs=True)
    layer_outputs = model_comp_h1_v1.layer_outputs

    l0_out = layer_outputs[0].detach() # layer 0 outputs
    l0_out = l0_out.cpu().detach().numpy()

    # convert data_y into rgb image data
    data_y_h1 = data_y_h1.squeeze(1)
    data_y_h2 = data_y_h2.squeeze(1)
    data_y_h1_grid = np.reshape(data_y_h1, [num_data_points, num_data_points, 1])
    data_y_rgb = np.concatenate([data_y_h1_grid, data_y_h1_grid, data_y_h1_grid], 2)

    # convert array into image
    img = Image.from_array(data_y_rgb)

    # image utils for easier viewing.
    img.background_color = Color('skyblue')
    img.virtual_pixel = 'background'
    dist_img = img.clone()
    print('l0: ', l0_out, torch.min(data_x), torch.max(data_x))


    bins = num_data_points
    # NOTE: here, first index = column and second index traverse rows

    # source and dest points for experiment
    source_points = (
        (0, 0),
        (0, bins),
        (bins, 0),
        (bins, bins)
    )
    destination_points = (
        (0,0),
        (0 ,bins+2),
        (bins, 0),
        (bins, bins)
    )

    # source points = input data.
    # destination points = input data after rkhs layer to see how this layer morph the input data.
    data_x = data_x.cpu().detach().numpy()
    data_x_new = remap(data_x, np.min(data_x), np.max(data_x), 0, num_data_points)
    l0_out = remap(l0_out, np.min(l0_out), np.max(l0_out), np.min(l0_out)*(num_data_points/2), np.max(l0_out)*(num_data_points/2)) + num_data_points/2
    source_points = data_x_new
    destination_points = l0_out

    print('data_x_new', data_x, np.min(l0_out), np.max(l0_out))

    # gives the effect of zoom and pan.
    dist_img.artifacts['distort:viewport'] = '500x500-250-250'

    # prepare the args and distorting.
    order = chain.from_iterable(zip(source_points, destination_points))
    dist_args = list(chain.from_iterable(order))
    dist_img.distort('bilinear_forward', dist_args)

    #plotting...
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(data_y_rgb)
    axs[0,1].imshow(dist_img)
    plt.show()


def repr_fig6(num_data_points=10, num_epochs=10000):
    # check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print('Using {} device'.format(device))

    _,ranges,data_x,data_y_h1, data_y_h2 = createSyntheticData(num_data_points)
    data_x = data_x.to(device)
    data_y_h1 = data_y_h1.to(device)
    data_y_h2 = data_y_h2.to(device)*1.0

    # calculating the models for constructing h1 and h2 functions.
    first_layer_poly_kernel_degree = 2
    model_comp_h1_v1 = e2eKRR(data_x, data_y_h1, ranges, first_layer_poly_kernel_degree, device, num_epochs, retain_layer_outputs=True)

    layer_outputs = model_comp_h1_v1.layer_outputs
    #print(layer_outputs)

    l0 = layer_outputs[0].detach()
    data_y_h1 = data_y_h1.squeeze(1)
    data_y_h2 = data_y_h2.squeeze(1)

    #data_y_h1 = data_y_h1.reshape([10, 10])
    mask = torch.abs(data_x - l0).less(1)
    mask = mask.sum(1).bool()
    masked_y_h1 = data_y_h1*mask


    l0_grid = l0.reshape([ int(data_x.shape[0]**(1/2)), int(data_x.shape[0]**(1/2)), data_x.shape[1]])

    print(data_y_h1.shape, data_y_h2.shape, data_y_h2.reshape(10, 10))

    data_x_grid = data_x.reshape([ int(data_x.shape[0]**(1/2)), int(data_x.shape[0]**(1/2)), data_x.shape[1]])
    dx1 = dx2 = data_x_grid[:, :, 1][:1, :].squeeze().cpu().detach().numpy()
    data_y_h1 = data_y_h1.reshape(data_x_grid.shape[:2])
    data_y_h1 = data_y_h1.cpu().detach().numpy() 
    data_y_h2 = data_y_h2.reshape(data_x_grid.shape[:2])
    data_y_h2 = data_y_h2.cpu().detach().numpy() 
    masked_y_h1 = masked_y_h1.reshape(data_x_grid.shape[:2])
    masked_y_h1 = masked_y_h1.cpu().detach().numpy() 


    print("dx1: ", dx1,"dx2: ", dx2,"data_y_h2: ", data_y_h2, 'masked_y1: ', masked_y_h1, 'l0: ', l0, l0_grid)
    #data_y_h2 = torch.ones_like(data_y_h2)

    fig, axs = plt.subplots(2,2)

    img = data_y_h1.copy()
    A = img.shape[0] / 3.0
    w = 2.0 / img.shape[1]


    shift = lambda x: A * np.sin(2.0*np.pi*x * w)

    print(img[:,0], img.shape)

    for i in range(img.shape[0]):
        img[:,i] = np.roll(img[:,i], int(shift(i)))

    axs[0,0].imshow(data_y_h1)
    axs[0,1].imshow(data_y_h2)
    axs[1,0].imshow(masked_y_h1)
    axs[1,1].imshow(img)


    plt.subplot_tool()
    #plt.show()

    bins = num_data_points

    im = Image.fromarray(data_y_h1)
    dst_grid = griddify(shape_to_rect(im.size), bins - 1, bins -1)

    l0min = l0.min(0).values
    l0max = l0.max(0).values

    linspace_x = np.linspace(l0min[0], l0max[0], bins)
    linspace_y = np.linspace(l0min[1], l0max[1], bins)
    catgor_x = np.digitize(l0[:, 0], linspace_x)
    catgor_y = np.digitize(l0[:, 1], linspace_y)
    catgor_x = np.expand_dims(catgor_x, 1)
    catgor_y = np.expand_dims(catgor_y, 1)

    print(linspace_x.shape, catgor_x.shape, catgor_y.shape)
    catgor = np.concatenate([catgor_x, catgor_y], 1)
    catgor_grid = catgor.reshape([ int(bins**(1)), int(bins**(1)), data_x.shape[1]])

    # converting category to index
    index_grid = catgor_grid - 1
    index_grid = np.abs((index_grid >= 0)*index_grid)

    print('min&max: ', torch.min(l0_grid), torch.max(l0_grid), catgor.shape, np.min(dst_grid), dst_grid.shape, index_grid.shape)

    # TODO: Convert category to index and map them to dst_grid.
    # this will give us the grid coordinates of the transformed( transformed by our network layer ) input data grid.

    #src_grid = distort_grid(dst_grid, 1)

    mesh = grid_to_mesh(index_grid, dst_grid)

    #print('dst_grid: ', dst_grid, 'src_grid: ', src_grid, 'mesh: ', mesh, len(mesh), len(mesh[8]), len(mesh[8][0]))
    im = im.transform(im.size, Image.MESH, mesh)

    axs[1,1].imshow(np.array(im))
    plt.show()
