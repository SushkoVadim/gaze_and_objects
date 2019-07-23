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
from convNet import scoreNet
from triple_dataset import tripleDataset
from torch.autograd import Variable
import os
from my_resnet18 import resnet18

root_path = "/local_scratch/vsushko/heatmaps_compare/"
data_descr_path = root_path+"data_descript"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_of_epoch = 400

net = scoreNet()

net = net.to(device)
net.train()
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


if os.path.isfile('right_config.npy'):
    start_array = np.load('right_config.npy')
    start_epoch = start_array[0]
    start_iter = start_array[1]
    not_already_configured_epoch = True
    not_already_configured_iter  = True
    net.load_state_dict(torch.load("model_trained/right_fc_trained_model.ptch"))
else:
    print('no config file')
    start_epoch = 0
    start_iter = 0
    not_already_configured_epoch = False
    not_already_configured_iter  = False


print(start_epoch, start_iter)
print("started")

# DATA PREPARATION
for param in net.part_face.parameters():
    param.requires_grad = False
for param in net.part_obj.parameters():
    param.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

pic_dataset = tripleDataset(descr_file = data_descr_path+'/train_annotations.txt')

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

        if i == 9918 or i == 3118:
            continue

        full_picture, object_picture, negative_obj, negative_ground, face_picture, ground_truth_gaze, ground_truth_eyes = pic_dataset[i]

        full_picture = full_picture.to(device)
        object_picture = object_picture.to(device)
        negative_obj = negative_obj.to(device)
        negative_ground = negative_ground.to(device)
        face_picture = face_picture.to(device)
        ground_truth_gaze = ground_truth_gaze.to(device)
        ground_truth_eyes = ground_truth_eyes.to(device)

        output = net(full_picture, face_picture, object_picture, ground_truth_gaze, ground_truth_eyes)
        loss1 = criterion(output, torch.tensor([1]).long().to("cuda:0"))
        net.zero_grad()
        loss1.backward()
        optimizer.step()

        output = net(full_picture, face_picture, negative_obj, negative_ground, ground_truth_eyes)
        loss2 = criterion(output, torch.tensor([0]).long().to("cuda:0"))
        net.zero_grad()
        loss2.backward()
        optimizer.step()

        #print(i, loss1.detach().item())
        #print(i, loss2.detach().item())

        running_loss1 += loss1.detach().item()
        running_loss2 += loss2.detach().item()
        if i % 100 == 99:
            print(epoch, i, running_loss1)
            print(epoch, i, running_loss2)
            print()
            running_loss1 = 0.0
            running_loss2 = 0.0
            torch.save(net.state_dict(), "model_trained/right_fc_trained_model.ptch")
            np.save("right_config", np.array([epoch, i]))


    print("finished", epoch)
    torch.save(net.state_dict(), "model_trained/right_fc_trained_model_after_"+str(epoch)+"epoch"+".ptch")


print('Finished Training')

