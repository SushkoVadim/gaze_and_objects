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
from convNet_noAlex import sumNet_noAlex
from gaze_dataset import gazeDataset
from torch.autograd import Variable
import os
from my_resnet18 import resnet18
import tensorflow as tf
from models import ResNet50UpProj

root_path = "/local_scratch/vsushko/heatmaps_depth/"
data_descr_path = root_path+"data_descript"
model_depth_path = "checkpoints/NYU_FCRN.ckpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_of_epoch = 40

net = sumNet_noAlex()
#net.load_state_dict(torch.load("data_descript/gaze_trained_model_after_0epoch.ptch", map_location='cpu'))
previous_first_layer = net.resnet2.conv1.weight.data
net.resnet2.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.resnet2.conv1.weight.data[:, 0:3, :, :] = previous_first_layer


net = net.to(device)
net.train()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


if os.path.isfile('config.npy'):
    start_array = np.load('config.npy')
    start_epoch = start_array[0]
    start_iter = start_array[1]
    not_already_configured_epoch = True
    not_already_configured_iter  = True
    net.load_state_dict(torch.load("model_trained/depth_trained_model.ptch"))
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
description = pd.read_csv(data_descr_path+'/train_annotations.txt', index_col=False, header=None)

#tensorflow stuff
tf_height = 228
tf_width = 304
tf_channels = 3
tf_batch_size = 1
tf.reset_default_graph()
tf_input_node = tf.placeholder(tf.float32, shape=(None, tf_height, tf_width, tf_channels))
tf_net_depth = ResNet50UpProj({'data': tf_input_node}, tf_batch_size, 1, False)
tf_saver = tf.train.Saver()


trainloader = torch.utils.data.DataLoader(pic_dataset, batch_size=1, shuffle=False)
print("ready to start first epoch")

with tf.Session() as sess:
    tf_saver.restore(sess, model_depth_path)
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

            img_path = description.iloc[i, 0]
            img = Image.open("/local_scratch/vsushko/heatmaps_compare/"+img_path)
            if not len(np.array(img).shape) == 3:
                continue
            img = img.resize([tf_width, tf_height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis=0)
            depth_map = sess.run(tf_net_depth.get_output(), feed_dict={tf_input_node: img})[0, :, :, 0]
            depth_map = cv2.resize(depth_map, (image.shape[3], image.shape[2]), interpolation=cv2.INTER_NEAREST)
            face_and_depth = torch.cat((face, torch.FloatTensor(depth_map).view(1, 1, depth_map.shape[0], depth_map.shape[1])), dim=1)

            full_picture = image.to(device)
            face_and_depth = face_and_depth.to(device)
            ground_truth_gauss = ground.to(device)

            outputs = net(full_picture, face_and_depth)
            loss = criterion(outputs[0][0, 0, :, :].view(-1), ground_truth_gauss.view(-1))
            net.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach_().item()
            if i % 500 == 499:
                print(epoch, i, running_loss)
                print()
                running_loss = 0.0
                torch.save(net.state_dict(), "model_trained/depth_trained_model.ptch")
                np.save("config", np.array([epoch, i]))


        print("finished", epoch)
        torch.save(net.state_dict(), "model_trained/depth_trained_model_after_"+str(epoch)+"epoch"+".ptch")


print('Finished Training')

