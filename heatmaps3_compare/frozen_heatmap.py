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


if os.path.isfile('frozen_config.npy'):
    start_array = np.load('frozen_config.npy')
    start_epoch = start_array[0]
    start_iter = start_array[1]
    not_already_configured_epoch = True
    not_already_configured_iter  = True
    net.load_state_dict(torch.load("model_trained/frozen_triple_trained_model.ptch"))
else:
    print('no config file')
    start_epoch = 0
    start_iter = 0
    not_already_configured_epoch = False
    not_already_configured_iter  = False


print(start_epoch, start_iter)
print("started")

for param in net.resnet2.parameters():
    param.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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

    running_loss = 0.0
    for i in range(len(pic_dataset)):

        if not_already_configured_iter:
            if i < start_iter:
                continue
            else:
                not_already_configured_iter = False

        if i == 9918:
            continue

        full_picture, object_picture, face_picture, ground_truth_gauss = pic_dataset[i]

        full_picture = full_picture.to(device)
        object_picture = object_picture.to(device)
        face_picture = face_picture.to(device)
        ground_truth_gauss = ground_truth_gauss.to(device)

        outputs = net(full_picture, face_picture, object_picture)

        #plt.imshow(outputs[0].cpu().detach().numpy()[0, 0, :, :], cmap='hot', interpolation='nearest')
        #plt.show()

        loss = criterion(outputs[0][0, 0, :, :].view(-1), ground_truth_gauss.view(-1))

        net.zero_grad()
        loss.backward()
        optimizer.step()

        #print(i, loss.detach_().item())

        running_loss += loss.detach_().item()
        if i % 500 == 499:
            print(epoch, i, running_loss)
            print()
            running_loss = 0.0
            torch.save(net.state_dict(), "model_trained/frozen_triple_trained_model.ptch")
            np.save("frozen_config", np.array([epoch, i]))


    print("finished", epoch)
    torch.save(net.state_dict(), "model_trained/frozen_triple_trained_model_after_"+str(epoch)+"epoch"+".ptch")


print('Finished Training')

