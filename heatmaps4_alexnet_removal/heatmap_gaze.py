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
from convNet import sumNet
from gaze_dataset import gazeDataset
from torch.autograd import Variable
import os
from my_resnet18 import resnet18

root_path = "/local_scratch/vsushko/heatmaps_with_YOLO/"
data_descr_path = root_path+"data_descript"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_of_epoch = 40

net = sumNet()

net = net.to(device)
net.train()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


if os.path.isfile('config_gaze.npy'):
    start_array = np.load('config_gaze.npy')
    start_epoch = start_array[0]
    start_iter = start_array[1]
    not_already_configured_epoch = True
    not_already_configured_iter  = True
    net.load_state_dict(torch.load("model_trained/gaze_trained_model.ptch"))
else:
    print('no config file')
    start_epoch = 0
    start_iter = 0
    not_already_configured_epoch = False
    not_already_configured_iter  = False


print(start_epoch, start_iter)
print("started")

# DATA PREPARATION

#def gen_gauss(pic, gaze, sigma):
#    x_grid, y_grid = np.meshgrid(np.linspace(1, pic.shape[1], pic.shape[1]), np.linspace(1,pic.shape[0], pic.shape[0]))
#    mu = np.array([gaze[1]*pic.shape[1], gaze[0]*pic.shape[0]])
#    d = np.sqrt((x_grid-mu[0])*(x_grid-mu[0])+(y_grid-mu[1])*(y_grid-mu[1]))
#    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
#    #g = 1*(d < 20)
#    g = g/np.max(g)
#    return g

pic_dataset = gazeDataset(descr_file = data_descr_path+'/train_annotations.txt')

trainloader = torch.utils.data.DataLoader(pic_dataset, batch_size=1, shuffle=False)
print("ready to start first epoch")

for epoch in range(num_of_epoch):
# TRAINING
    if not_already_configured_epoch:
        if epoch < start_epoch:
            continue
        else:
            not_already_configured_epoch = False

    running_loss = 0.0
    for i in range(len(pic_dataset)):

        if not_already_configured_iter:
            if i < start_iter:
                continue
            else:
                not_already_configured_iter = False

        if i == 9918:
            continue

        image, face, ground = pic_dataset[i]

        full_picture = image.to(device)
        face_picture = face.to(device)
        ground_truth_gauss = ground.to(device)

        outputs = net(full_picture, face_picture)
        loss = criterion(outputs[0][0, 0, :, :].view(-1), ground_truth_gauss.view(-1))
        net.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.detach_().item()
        if i % 500 == 499:
            print(epoch, i, running_loss)
            print()
            running_loss = 0.0
            torch.save(net.state_dict(), "model_trained/gaze_trained_model.ptch")
            np.save("config_gaze", np.array([epoch, i]))


    print("finished", epoch)
    torch.save(net.state_dict(), "model_trained/gaze_trained_model_after_"+str(epoch)+"epoch"+".ptch")


print('Finished Training')
