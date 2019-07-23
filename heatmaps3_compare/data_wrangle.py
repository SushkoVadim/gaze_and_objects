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


descr_file = '/local_scratch/vsushko/heatmaps_compare/data_descript/train_annotations.txt'
root_folder = '/local_scratch/vsushko/heatmaps_compare/'
obj_path = '/local_scratch/vsushko/heatmaps_compare/train_obj_alpha/'
model_path = '/local_scratch/vsushko/heatmaps_compare/model'
description = pd.read_csv(descr_file, header=None)

mat = sio.loadmat(model_path+'/places_mean_resize.mat')['image_mean']
image_mean1 = sio.loadmat(model_path+'/imagenet_mean_resize.mat')['image_mean']

for idx in range(0, len(description)):

    img_path = description.iloc[idx, 0]
    gaze = (description.iloc[idx, 8], description.iloc[idx, 9])
    eyes = (description.iloc[idx, 6], description.iloc[idx, 7])
    image = cv2.imread(root_folder+img_path)
    pic = image
    #obj = cv2.imread(obj_path+'face'+str(idx)+'.jpg')
    ground_truth_eyes = gen_gauss(image, eyes)
    ground_truth_gaze = gen_gauss(image, gaze)

    alpha = 0.3
    w_x = int(np.floor(alpha * pic.shape[1]))
    w_y = int(np.floor(alpha * pic.shape[0]))
    if w_x % 2 == 0:
        w_x += 1
    if w_y % 2 == 0:
        w_y += 1
    im_face = np.zeros((pic.shape[0], pic.shape[1], 3), dtype='uint8')
    center = np.floor([gaze[0] * pic.shape[1], gaze[1] * pic.shape[0]]).astype(int)
    d_x = int(np.floor((w_x - 1) / 2))
    d_y = int(np.floor((w_y - 1) / 2))
    tmp = center[0] - d_x
    img_l = max(0, tmp)
    delta_l = max(0, -tmp)
    tmp = center[0] + d_x + 1
    img_r = min(pic.shape[1], tmp)
    delta_r = w_x - (tmp - img_r)
    tmp = center[1] - d_y
    img_t = max(0, tmp)
    delta_t = max(0, -tmp)
    tmp = center[1] + d_y + 1
    img_b = min(pic.shape[0], tmp)
    delta_b = w_y - (tmp - img_b)
    im_face[img_t:img_b, img_l:img_r, :] = pic[img_t:img_b, img_l:img_r, :]
    face_numpy = np.array(im_face)
    face_pic = Image.fromarray(face_numpy)
    obj = face_pic
# --------------------------------------------------------
    alpha = 0.3
    w_x = int(np.floor(alpha * pic.shape[1]))
    w_y = int(np.floor(alpha * pic.shape[0]))
    if w_x % 2 == 0:
        w_x += 1
    if w_y % 2 == 0:
        w_y += 1
    im_face = np.zeros((pic.shape[0], pic.shape[1], 3), dtype='uint8')
    center = np.floor([eyes[0] * pic.shape[1], eyes[1] * pic.shape[0]]).astype(int)
    d_x = int(np.floor((w_x - 1) / 2))
    d_y = int(np.floor((w_y - 1) / 2))
    tmp = center[0] - d_x;
    img_l = max(0, tmp);
    delta_l = max(0, -tmp);
    tmp = center[0] + d_x + 1;
    img_r = min(pic.shape[1], tmp);
    delta_r = w_x - (tmp - img_r);
    tmp = center[1] - d_y;
    img_t = max(0, tmp);
    delta_t = max(0, -tmp);
    tmp = center[1] + d_y + 1;
    img_b = min(pic.shape[0], tmp);
    delta_b = w_y - (tmp - img_b);
    im_face[img_t:img_b, img_l:img_r, :] = pic[img_t:img_b, img_l:img_r, :]
    face_numpy = np.array(im_face)
    face_pic = Image.fromarray(face_numpy)
    face = face_pic

    image_mean_tmp = cv2.resize(mat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    image = image - image_mean_tmp

    image = np.transpose(image, (2, 0, 1))
    obj = np.transpose(obj, (2, 0, 1))
    face = np.transpose(face, (2, 0, 1))

    dim = image.shape
    image = torch.tensor(image).view(1, 3, dim[1], dim[2]).float()
    obj = torch.tensor(obj).view(1, 3, dim[1], dim[2]).float()
    face = torch.tensor(face).view(1, 3, dim[1], dim[2]).float()
    ground_truth_eyes = torch.tensor(ground_truth_eyes).view(1, dim[1], dim[2]).float()
    ground_truth_gaze = torch.tensor(ground_truth_gaze).view(1, dim[1], dim[2]).float()

    os.makedirs(os.path.dirname("torch_data_trainannotations/images_torch1/"+"{:03d}".format(int(idx/1000))+"/image_torch"+"{:06d}".format(idx)), exist_ok=True)
    os.makedirs(
        os.path.dirname("torch_data_trainannotations/obj_torch1/"+"{:03d}".format(int(idx/1000))+"/obj_torch"+"{:06d}".format(idx)),
        exist_ok=True)
    os.makedirs(
        os.path.dirname("torch_data_trainannotations/face_torch1/" + "{:03d}".format(
            int(idx / 1000)) + "/face_torch" + "{:06d}".format(idx)),
        exist_ok=True)
    os.makedirs(
        os.path.dirname("torch_data_trainannotations/grounds_torch_eyes1/" + "{:03d}".format(
            int(idx / 1000)) + "/ground_torch_eyes" + "{:06d}".format(idx)),
        exist_ok=True)
    os.makedirs(
        os.path.dirname("torch_data_trainannotations/grounds_torch_gaze1/"+"{:03d}".format(int(idx/1000))+"/ground_torch_gaze" + "{:06d}".format(idx)),
        exist_ok=True)
    torch.save(image, "torch_data_trainannotations/images_torch1/"+"{:03d}".format(int(idx/1000))+"/image_torch"+"{:06d}".format(idx))
    torch.save(obj, "torch_data_trainannotations/obj_torch1/"+"{:03d}".format(int(idx/1000))+"/obj_torch"+"{:06d}".format(idx))
    torch.save(face, "torch_data_trainannotations/face_torch1/" + "{:03d}".format(int(idx / 1000)) + "/face_torch" + "{:06d}".format(idx))
    torch.save(ground_truth_eyes, "torch_data_trainannotations/grounds_torch_eyes1/"+"{:03d}".format(int(idx/1000))+"/ground_torch_eyes" + "{:06d}".format(idx))
    torch.save(ground_truth_gaze, "torch_data_trainannotations/grounds_torch_gaze1/" + "{:03d}".format(int(idx / 1000)) + "/ground_torch_gaze" + "{:06d}".format(idx))

    if idx % 500 == 0:
        print(idx)
