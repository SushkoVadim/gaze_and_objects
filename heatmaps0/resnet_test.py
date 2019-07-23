import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw
import scipy.io as sio
import cv2
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import sys
from resnet18 import resnet18, BasicBlock

good_old_path = "/local_scratch/vsushko/AAnaconda/02-25 transfer from caffe/"

data0 = list()
input_file = open("/local_scratch/vsushko/AAnaconda/02-25 transfer from caffe/data_descript/train_annotations.txt", 'r')

for num_of_line in range(0, 1):

    corn = [0.0, 0.0]
    leng = [0.0, 0.0]
    eyes = [0.0, 0.0]
    gaze = [0.0, 0.0]
    [file_train, idd, corn[0], corn[1], leng[0], leng[1], eyes[0], eyes[1], gaze[0],
     gaze[1], _, _] = input_file.readline().split(",")
    corn = list(map(float, corn))
    leng = list(map(float, leng))
    eyes = list(map(float, eyes))
    gaze = list(map(float, gaze))

    pic = cv2.imread(good_old_path + file_train)
    mat = sio.loadmat(good_old_path + 'model/places_mean_resize.mat')['image_mean']
    image_mean = cv2.resize(mat, (pic.shape[1], pic.shape[0]), interpolation=cv2.INTER_CUBIC)

    transformed_pic = pic - image_mean
    transformed_pic = np.transpose(transformed_pic, (2, 0, 1))
    # ----------------------------------------------------------------------------
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
    # face_pic.show()
    mat1 = sio.loadmat(good_old_path + 'model/imagenet_mean_resize.mat')['image_mean']
    image_mean_faces = cv2.resize(mat1, (pic.shape[1], pic.shape[0]))
    transformed_face = im_face - image_mean_faces
    transformed_face = np.transpose(transformed_face, (2, 0, 1))

    full_picture = torch.tensor(transformed_pic).view(3, pic.shape[0], pic.shape[1]).float()
    face_picture = torch.tensor(transformed_face).view(3, pic.shape[0], pic.shape[1]).float()
    #ground_truth_gauss = torch.tensor(gen_gauss(pic, gaze)).view(1, pic.shape[0], pic.shape[1]).float()
    data0.append((full_picture, face_picture))
    # plt.imshow(ground_truth_gauss.detach().numpy()[0, :, :], cmap='hot', interpolation='nearest')
    # plt.show()
    print(pic.shape)


net = resnet18(pretrained=True)
trainloader = torch.utils.data.DataLoader(data0, batch_size=1, shuffle=True)
print(net(list(trainloader)[0][0]).shape)
