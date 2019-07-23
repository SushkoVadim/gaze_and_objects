import pandas as pd
import numpy as np
import cv2
import torch
import scipy.io as sio
import os

def gen_gauss(pic, eyes):
    x_grid, y_grid = np.meshgrid(np.linspace(1, pic.shape[1], pic.shape[1]), np.linspace(1,pic.shape[0], pic.shape[0]))
    sigma = 20
    mu = np.array([eyes[0]*pic.shape[1], eyes[1]*pic.shape[0]])
    d = np.sqrt((x_grid-mu[0])*(x_grid-mu[0])+(y_grid-mu[1])*(y_grid-mu[1]))
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    #g = 1*(d < 20)
    g = g/np.max(g)
    return g


descr_file = '/local_scratch/vsushko/heatmaps_with_YOLO/data_descript/train_annotations.txt'
root_folder = '/local_scratch/vsushko/heatmaps_with_YOLO/'
obj_path = '/local_scratch/vsushko/heatmaps_with_YOLO/train_obj_alpha/'
model_path = '/local_scratch/vsushko/heatmaps_with_YOLO/model'
description = pd.read_csv(descr_file, header=None)

image_mean = sio.loadmat(model_path+'/places_mean_resize.mat')['image_mean']
image_mean1 = sio.loadmat(model_path+'/imagenet_mean_resize.mat')['image_mean']

for idx in range(0, len(description)):

    img_path = description.iloc[idx, 0]
    #obj_num = img_path[15:]
    gaze = (description.iloc[idx, 8], description.iloc[idx, 9])
    eyes = (description.iloc[idx, 6], description.iloc[idx, 7])
    image = cv2.imread(root_folder+img_path)
    obj = cv2.imread(obj_path+'face'+str(idx)+'.jpg')
    ground_truth = gen_gauss(image, eyes)

    image_mean_tmp = cv2.resize(image_mean, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    image = image - image_mean_tmp
    image = np.transpose(image, (2, 0, 1))

    obj = np.transpose(obj, (2, 0, 1))

    dim = image.shape
    image = torch.tensor(image).view(3, dim[1], dim[2]).float()
    obj = torch.tensor(obj).view(3, dim[1], dim[2]).float()
    ground_truth = torch.tensor(ground_truth).view(1, dim[1], dim[2]).float()

    os.makedirs(os.path.dirname("torch_data/images_torch1/"+"{:03d}".format(int(idx/1000))+"/image_torch"+"{:06d}".format(idx)), exist_ok=True)
    os.makedirs(
        os.path.dirname("torch_data/obj_torch1/"+"{:03d}".format(int(idx/1000))+"/obj_torch"+"{:06d}".format(idx)),
        exist_ok=True)
    os.makedirs(
        os.path.dirname("torch_data/grounds_torch1/"+"{:03d}".format(int(idx/1000))+"/ground_torch" + "{:06d}".format(idx)),
        exist_ok=True)
    torch.save(image, "torch_data/images_torch1/"+"{:03d}".format(int(idx/1000))+"/image_torch"+"{:06d}".format(idx))
    torch.save(obj, "torch_data/obj_torch1/"+"{:03d}".format(int(idx/1000))+"/obj_torch"+"{:06d}".format(idx))
    torch.save(ground_truth, "torch_data/grounds_torch1/"+"{:03d}".format(int(idx/1000))+"/ground_torch" + "{:06d}".format(idx))

    if idx % 500 == 0:
        print(idx)
