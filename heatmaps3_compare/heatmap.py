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
import matplotlib.image as mpimg
import time
import sys
from convNet import sumNet3
from triple_dataset import tripleDataset
from torch.autograd import Variable
import os
from my_resnet18 import resnet18

root_path = "/local_scratch/vsushko/heatmaps_compare/"
data_descr_path = root_path+"data_descript"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_of_epoch = 40

net = sumNet3()

net = net.to(device)
net.train()
criterion = nn.BCELoss()

# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


if os.path.isfile('config.npy'):
    start_array = np.load('config.npy')
    start_epoch = start_array[0]
    start_iter = start_array[1]
    not_already_configured_epoch = True
    not_already_configured_iter  = True
    net.load_state_dict(torch.load("model_trained/triple_trained_model.ptch"))
else:
    print('no config file')
    start_epoch = 0
    start_iter = 0
    not_already_configured_epoch = False
    not_already_configured_iter  = False


print(start_epoch, start_iter)
print("started")

# DATA PREPARATION
optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

pic_dataset = tripleDataset(descr_file = data_descr_path+'/train_annotations.txt')

trainloader = torch.utils.data.DataLoader(pic_dataset, batch_size=1, shuffle=False)
print("ready to start first epoch")

for epoch in range(num_of_epoch):
# TRAINING
    if not_already_configured_epoch:
        if epoch < start_epoch:
            continue
        else:
            not_already_configured_epoch = False

    running_loss1 = 0.0
    running_loss2 = 0.0
    for i in range(len(pic_dataset)):

        if not_already_configured_iter:
            if i < start_iter:
                continue
            else:
                not_already_configured_iter = False

        if i == 9918:
            continue

        full_picture, object_picture, face_picture, ground_truth_gaze, ground_truth_eyes = pic_dataset[i]

        full_picture = full_picture.to(device)
        object_picture = object_picture.to(device)
        face_picture = face_picture.to(device)
        ground_truth_gaze = ground_truth_gaze.to(device)
        ground_truth_eyes = ground_truth_eyes.to(device)

        output_obj, _ = net(full_picture, face_picture, object_picture)

        loss1 = criterion(output_obj[0, 0, :, :].view(-1), ground_truth_eyes.view(-1))
        net.zero_grad()
        loss1.backward()
        optimizer.step()
        #print(i, loss1.detach().item())


        _, output_face = net(full_picture, face_picture, object_picture)

        loss2 = criterion(output_face[0, 0, :, :].view(-1), ground_truth_gaze.view(-1))
        #net.zero_grad()
        loss2.backward()
        optimizer.step()
        #print(i, loss2.detach().item())

        running_loss1 += loss1.detach().item()
        running_loss2 += loss2.detach().item()
        if i % 500 == 499:
            print(epoch, i, running_loss1)
            print(epoch, i, running_loss2)
            print()
            running_loss1 = 0.0
            running_loss2 = 0.0
            torch.save(net.state_dict(), "model_trained/triple_trained_model.ptch")
            np.save("config", np.array([epoch, i]))


    print("finished", epoch)
    torch.save(net.state_dict(), "model_trained/triple_trained_model_after_"+str(epoch)+"epoch"+".ptch")


print('Finished Training')

