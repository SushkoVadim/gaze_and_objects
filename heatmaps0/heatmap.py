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
from convNet import convNet
from face_dataset import picDataset
from torch.autograd import Variable

root_path = "/local_scratch/vsushko/"
path_test = root_path+"test"
data_descr_path = root_path+"data_descript"
path_train = root_path+"train"
model_path = root_path+"model"
use_gpu = True
sigma = 20
size_of_train = 10
num_of_epoch = 100

net = convNet()
net.load_state_dict(torch.load("model.ptch"))
net.train()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

if use_gpu:
    net = net.cuda()

# DATA PREPARATION

#def gen_gauss(pic, gaze, sigma):
#    x_grid, y_grid = np.meshgrid(np.linspace(1, pic.shape[1], pic.shape[1]), np.linspace(1,pic.shape[0], pic.shape[0]))
#    mu = np.array([gaze[1]*pic.shape[1], gaze[0]*pic.shape[0]])
#    d = np.sqrt((x_grid-mu[0])*(x_grid-mu[0])+(y_grid-mu[1])*(y_grid-mu[1]))
#    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
#    #g = 1*(d < 20)
#    g = g/np.max(g)
#    return g


for epoch in range(num_of_epoch):

    input_file = open(data_descr_path + "/train_annotations.txt")

    for data_part in range(125):

        data0 = list()

        for num_of_line in range(0, size_of_train):

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

            pic = cv2.imread(root_path + file_train)
            mat = sio.loadmat(model_path+"/places_mean_resize.mat")['image_mean']
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
            #face_numpy = np.array(im_face)
            #face_pic = Image.fromarray(face_numpy)
            #face_pic.show()
            mat1 = sio.loadmat(model_path+"/imagenet_mean_resize.mat")['image_mean']
            image_mean_faces = cv2.resize(mat1, (pic.shape[1], pic.shape[0]))
            transformed_face = im_face - image_mean_faces
            transformed_face = np.transpose(transformed_face, (2, 0, 1))

            full_picture = torch.tensor(transformed_pic).view(3, pic.shape[0], pic.shape[1]).float()
            face_picture = torch.tensor(transformed_face).view(3, pic.shape[0], pic.shape[1]).float()
            ground_truth_gauss = torch.tensor(gen_gauss(pic, gaze, sigma)).view(1, pic.shape[0], pic.shape[1]).float()
            data0.append((full_picture, face_picture, ground_truth_gauss))
            # plt.imshow(ground_truth_gauss.detach().numpy()[0, :, :], cmap='hot', interpolation='nearest')
            # plt.show()

        trainloader = torch.utils.data.DataLoader(data0, batch_size=1, shuffle=False)

        print(len(data0), data_part)

        # TRAINING

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs

            full_picture, face_picture, ground_truth_gauss = data
            full_picture = Variable(full_picture.cuda(), requires_grad = False)
            face_picture = Variable(face_picture.cuda(), requires_grad=False)
            ground_truth_gauss = Variable(ground_truth_gauss.cuda(), requires_grad=False)



            outputs = net(full_picture, face_picture)
            if i == 0:
                print('before', torch.cuda.max_memory_allocated(0))

            # plt.imshow(outputs[0].detach().numpy()[0, 0, :, :], cmap='hot', interpolation='nearest')
            # plt.show()

            loss = criterion(outputs[0][0, 0, :, :].view(-1), ground_truth_gauss.view(-1))

            # print(list(net.parameters())[5])
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # print(i, outputs)
            # print()

            running_loss += loss.detach_().item()
            if i % 1000 == 999:
                print(epoch, i, running_loss)
                running_loss = 0.0

        del trainloader
        torch.cuda.empty_cache()
    print("finished", epoch, data_part)
    torch.save(net.state_dict(), "trained_model.ptch")

print('Finished Training')


# TESTING

#trainloader1 = torch.utils.data.DataLoader(data0, batch_size=1,shuffle=True)
#dataiter = iter(trainloader1) #iterator of batches!

#corr = 0
#for i in range(4):
#    full_picture, face_picture, ground_truth_gauss = next(dataiter)
#    outputs = net(full_picture, face_picture)

#    plt.imshow(outputs[0].detach().numpy()[0, 0, :, :], cmap='hot', interpolation='nearest')
#    plt.show()

    #plt.imshow(ground_truth_gauss.detach().numpy()[0, 0, :, :], cmap='hot', interpolation='nearest')
    #plt.show()

    #loss = criterion(outputs[0][0, 0, :, :].view(-1), ground_truth_gauss.view(-1))

    #print(loss)
