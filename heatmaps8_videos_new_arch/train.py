import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from convNet_videos import sumNet_videos
import cv2
import math
import sklearn.metrics
import torch.optim as optim
import matplotlib.pyplot as plt
from video_dataset import video_dataset
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

def float2grid(input,side):
    b_s = input.size(0)
    output = torch.zeros(b_s,side*side)
    for j in range(b_s):
        if input[j,1] >=0 and input[j,1] <=1 and input[j,0] >=0 and input[j,0] <=1:
            class_final = np.rint(np.floor(input[j,1]*side)*side+np.floor(input[j, 0]*side)).numpy()
        else:
            class_final = side*side
        if(class_final < side*side):
            output[j,class_final] =1
    output = output.view(-1,1,side,side)
    return output

def mask_predict(output):
    output = np.array(output.detach().numpy())
    size = output.shape
    listx = np.linspace(0, size[0]-1, num=14, endpoint=True)
    listy = np.linspace(0, size[1]-1, num=14, endpoint=True)
    listx = list(map(int, (map(round, listx))))
    listy = list(map(int, (map(round, listy))))
    answer = np.zeros((13, 13))
    for i in range(13):
        for j in range(13):
            answer[i, j] = np.mean(output[ listx[i]:listx[i+1], listy[j]:listy[j+1] ])
    return answer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = sumNet_videos()

def validate(trainloader_test, net, device, epoch):
    avg_pa_sum0 = 0
    avg_augsum0 = 0
    sum_roc     = 0
    num_of_it0  = 0
    num_of_it1  = 0
    print("VALIDATION:")
    with torch.no_grad():
        array_il = np.array([])
        array_sm = np.array([])
        for i, obj in enumerate(trainloader_test):
            source_image, target_image, head_image, is_looked, ground, eyes, gaze = obj

            source_image = source_image.to(device)
            target_image = target_image.to(device)
            head_image = head_image.to(device)



            output, sigmoid = net(source_image, target_image, head_image)


            for num in range(output.shape[0]):
                if is_looked[num] == 1:
                    mask_gaze = np.zeros((169, 1, 1))
                    ffx = int(np.floor(gaze[0][num] * 13))
                    ffy = int(np.floor(gaze[1][num] * 13))
                    mask_gaze[13 * ffy + ffx, 0, 0] = 1
                    answer = mask_predict(output[0, 0, :, :].cpu())

                    cur_roc = roc_auc_score(mask_gaze.reshape(-1), answer.reshape(-1))
                    sum_roc += cur_roc
                    num_of_it1 += 1


            array_il = np.concatenate((array_il, is_looked.cpu()))
            array_sm = np.concatenate((array_sm, sigmoid[:, 1].view(-1).detach().cpu().numpy()))


            if i > 25:
                break



    avg_pa_sum0 += average_precision_score(array_il, array_sm)
    avg_augsum0 += roc_auc_score(array_il, array_sm)
    num_of_it0 += 1

    print("ROC GAZE:", sum_roc/num_of_it1)
    print("AVERAGE PRECISION:", avg_pa_sum0/num_of_it0)
    print("ROC SELECT:", avg_augsum0/num_of_it0)
    if epoch > 0:
        np.save("model_trained/validation"+str(epoch), np.array([sum_roc/num_of_it1, avg_pa_sum0/num_of_it0, avg_augsum0/num_of_it0]))

#state = torch.load("trained_video_after1.ptch", map_location=device)
#net.load_state_dict(state)

net.to(device)


test_file = "test_file.txt"
train_file = "train_file.txt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cur_folder = "/local_scratch/vsushko/heatmaps7_videos/data/"

criterion_select = nn.CrossEntropyLoss()
criterion_gaze   = nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

if os.path.isfile('video_config1.npy'):
    start_array = np.load('video_config1.npy')
    start_epoch = start_array[0]
    start_iter = start_array[1]
    not_already_configured_epoch = True
    not_already_configured_iter  = True
    net.load_state_dict(torch.load("model_trained/video_trained_model1.ptch"))
else:
    print('no config file')
    start_epoch = 0
    start_iter = 0
    not_already_configured_epoch = False
    not_already_configured_iter  = False

train_set = video_dataset("train_file.txt", cur_folder)
test_set  = video_dataset("test_file.txt", cur_folder)

bs = 5
trainloader_train = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
trainloader_test = torch.utils.data.DataLoader(test_set, batch_size=5, shuffle=True)

print(start_epoch, start_iter)
print("start!")

for epoch in range(100):
    # TRAINING
    if not_already_configured_epoch:
        if epoch < start_epoch:
            continue
        else:
            not_already_configured_epoch = False

    running_loss = 0.0
    for i, obj in enumerate(trainloader_train):
        source_image, target_image, head_image, is_looked, ground, eyes, gaze = obj

        source_image = source_image.to(device)
        target_image = target_image.to(device)
        head_image = head_image.to(device)
        is_looked = is_looked.to(device)
        ground = ground.to(device)

        out_gaze, out_select = net(source_image, target_image, head_image)

        loss1 = 6*criterion_gaze(out_gaze, ground)
        loss2 = criterion_select(out_select, is_looked)
        loss = loss1 + loss2

        net.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(epoch, i, running_loss, loss1.item(), loss2.item())
            running_loss = 0
            torch.save(net.state_dict(), "model_trained/video_trained_model1.ptch")
            np.save("video_config1", np.array([epoch, i]))
            validate(trainloader_test, net, device, 0)

    print("finished epoch")
    print(epoch, running_loss)
    validate(trainloader_test, net, device, epoch)
    print()
    running_loss = 0
    torch.save(net.state_dict(), "model_trained/video_trained_model_after1" + str(epoch) + ".ptch")




