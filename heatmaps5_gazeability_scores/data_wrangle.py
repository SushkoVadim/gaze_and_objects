import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
import scipy.io as sio
import os
import matplotlib.pyplot as plt


def gen_gauss(pic, eyes):
    x_grid, y_grid = np.meshgrid(np.linspace(1, pic.shape[1], pic.shape[1]), np.linspace(1,pic.shape[0], pic.shape[0]))
    sigma = 20
    mu = np.array([eyes[0]*pic.shape[1], eyes[1]*pic.shape[0]])
    d = np.sqrt((x_grid-mu[0])*(x_grid-mu[0])+(y_grid-mu[1])*(y_grid-mu[1]))
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    #g = 1*(d < 20)
    g = g/np.max(g)
    return g

description = pd.read_csv("centers_random.txt", header=None)

for idx in range(0, 125557):

    center = (description.iloc[idx, 1], description.iloc[idx, 2])

    obj = cv2.imread('negative_random_object_pictures/'+str(idx)+'.jpg')
    ground = gen_gauss(obj, (center[0], center[1]))

    obj = np.array(obj)
    obj = np.transpose(obj, (2, 0, 1))

    dim = obj.shape
    obj = torch.tensor(obj).view(1, 3, dim[1], dim[2]).float()

    os.makedirs(os.path.dirname("negative_random_object_tensors/"+"{:03d}".format(int(idx/1000))+"/neg_obj_torch"+"{:06d}".format(idx)), exist_ok=True)
    os.makedirs(os.path.dirname("negative_random_ground_tensors/"+"{:03d}".format(int(idx/1000))+"/neg_gro_torch"+"{:06d}".format(idx)), exist_ok=True)
    torch.save(obj, "negative_random_object_tensors/"+"{:03d}".format(int(idx/1000))+"/neg_obj_torch"+"{:06d}".format(idx))
    torch.save(ground, "negative_random_ground_tensors/"+"{:03d}".format(int(idx/1000))+"/neg_gro_torch"+"{:06d}".format(idx))

    if idx % 500 == 0:
        print(idx)
