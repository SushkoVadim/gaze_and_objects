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

sys.path.insert(0, '/local_scratch/vsushko/heatmaps_depth/deep-head-pose-master/code')
from hopenet import Hopenet
#-----------------------



head_pose_model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
saved_state_dict = torch.load(
    "/local_scratch/vsushko/heatmaps_depth/deep-head-pose-master/code/hopenet_robust_alpha1.pkl",
    map_location='cpu')
head_pose_model.load_state_dict(saved_state_dict)
head_pose_model.eval()

from test_on_video_dlib12 import head_pose
from utils import plot_pose_cube


def get_bbx(net_bbx, image, eyes):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net_bbx.setInput(blob)
    detections = net_bbx.forward()
    answer = list()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if eyes_in_bbox(eyes, detections[0, 0, i, 3:7]):
                answer.append((startX, startY, endX, endY))
    return answer


def eyes_in_bbox(eyes, detections):
    ans = False
    x1, y1, x2, y2 = detections
    if eyes[1] >= y1 and eyes[1] <= y2 and eyes[0] >= x1 and eyes[0] <= x2:
        ans = True
    return ans


def pic_to_eyes_coord(depth_map, eyes):
    angle_field_of_view = np.array(depth_map.shape) * 130 / max(depth_map.shape)
    angles0 = np.arange(depth_map.shape[0]) * angle_field_of_view[0] / depth_map.shape[0]
    angles1 = np.arange(depth_map.shape[1]) * angle_field_of_view[1] / depth_map.shape[1]
    angles0 = np.tile(angles0, (depth_map.shape[1], 1)).T / 180 * 3.14
    angles1 = np.tile(angles1, (depth_map.shape[0], 1)) / 180 * 3.14
    xx = depth_map * np.sin(angles0) * np.cos(angles1)
    yy = depth_map * np.sin(angles1)
    zz = np.sqrt(depth_map * depth_map - xx * xx - yy * yy)

    eyes_pix = np.array([eyes[0] * depth_map.shape[1], eyes[1] * depth_map.shape[0]]).astype(int)
    xx_eyes = xx[eyes_pix[1], eyes_pix[0]]
    yy_eyes = yy[eyes_pix[1], eyes_pix[0]]
    zz_eyes = zz[eyes_pix[1], eyes_pix[0]]
    return [(xx, yy, zz), (xx_eyes, yy_eyes, zz_eyes)]


def eyes_to_angle(whole_point_cloud, eyes_point, head_pose_vector):
    head_pose_vector = np.array(head_pose_vector) * 3.1415 / 180
    sight_dir = np.array([-np.sin(head_pose_vector[1]), -np.cos(head_pose_vector[1]) * np.sin(head_pose_vector[0]),
                          -np.cos(head_pose_vector[1]) * np.cos(head_pose_vector[0])])

    centered_xx = whole_point_cloud[0] - eyes_point[0] - 0.2 * sight_dir[0]
    centered_yy = whole_point_cloud[1] - eyes_point[1] - 0.2 * sight_dir[1]
    centered_zz = whole_point_cloud[2] - eyes_point[2] - 0.2 * sight_dir[2]

    norm = np.sqrt(centered_xx * centered_xx + centered_yy * centered_yy + centered_zz * centered_zz)
    cosina = (centered_xx * sight_dir[0] + centered_yy * sight_dir[1] + centered_zz * sight_dir[2]) / norm
    angles = np.arccos(cosina) * 180 / 3.1415
    return angles


def geometry_mask(angle_map):
    result = 0.0 * (angle_map >= 0)
    result += 1.0 * (angle_map < 30).astype(float)
    result += (angle_map > 30) * (angle_map < 60) * 0.6
    result += 0.3 * ((angle_map > 60) * (angle_map < 80))
    result += 0.1 * ((angle_map > 80) * (angle_map < 110))
    return result


#--------------------------

root_path = "/local_scratch/vsushko/heatmaps_depth/"
data_descr_path = root_path+"data_descript"
model_depth_path = "checkpoints/NYU_FCRN.ckpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_of_epoch = 40

net = sumNet_noAlex()
net.load_state_dict(torch.load("data_descript/gaze_trained_model_after_0epoch.ptch", map_location='cpu'))
previous_first_layer = net.resnet2.conv1.weight.data
net.resnet2.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.resnet2.conv1.weight.data[:, 0:3, :, :] = previous_first_layer


net = net.to(device)
net.train()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


if os.path.isfile('config_angles.npy'):
    start_array = np.load('config_angles.npy')
    start_epoch = start_array[0]
    start_iter = start_array[1]
    not_already_configured_epoch = True
    not_already_configured_iter  = True
    net.load_state_dict(torch.load("model_trained/angles_trained_model.ptch"))
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

            boxes = get_bbx(net_bbx, image, eyes)
            if len(boxes) > 0:
                head_pose_vector = head_pose(head_pose_model, image, boxes[0])
                whole_point_cloud, eyes_point = pic_to_eyes_coord(depth_map, eyes)
                angle_map = eyes_to_angle(whole_point_cloud, eyes_point, head_pose_vector)
                face_and_angles = torch.cat((face_and_depth, torch.FloatTensor(angle_map).view(1, 1, depth_map.shape[0], depth_map.shape[1])), dim=1)
                full_picture = image.to(device)
                face_and_angles = face_and_angles.to(device)
                ground_truth_gauss = ground.to(device)

                outputs = net(full_picture, face_and_angles)
                loss = criterion(outputs[0][0, 0, :, :].view(-1), ground_truth_gauss.view(-1))
                net.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.detach_().item()
            if i % 500 == 499:
                print(epoch, i, running_loss)
                print()
                running_loss = 0.0
                torch.save(net.state_dict(), "model_trained/angles_trained_model.ptch")
                np.save("config_angles", np.array([epoch, i]))


        print("finished", epoch)
        torch.save(net.state_dict(), "model_trained/angles_trained_model_after_"+str(epoch)+"epoch"+".ptch")


print('Finished Training')

