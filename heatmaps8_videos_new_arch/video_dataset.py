from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def gen_gauss(pic, eyes):
    x_grid, y_grid = np.meshgrid(np.linspace(1, pic.shape[1], pic.shape[1]), np.linspace(1,pic.shape[0], pic.shape[0]))
    sigma = 20
    mu = np.array([eyes[0]*pic.shape[1], eyes[1]*pic.shape[0]])
    d = np.sqrt((x_grid-mu[0])*(x_grid-mu[0])+(y_grid-mu[1])*(y_grid-mu[1]))
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    #g = 1*(d < 20)

    if np.max(g) != 0:
        g = g/np.max(g)
    return g

class video_dataset(Dataset):
    def __init__(self, descr_file, home_path):
        self.description = pd.read_csv(descr_file, index_col=False, header=None)
        self.home_path = home_path

    def __len__(self):
        return len(self.description)

    def __getitem__(self, idx):

        cur_idx = self.description.iloc[idx, 0]
        cur_num = self.description.iloc[idx, 1]
        source_name = self.description.iloc[idx, 2]
        target_name = self.description.iloc[idx, 3]
        head_name = self.description.iloc[idx, 4]
        flip = self.description.iloc[idx, 5]
        eyes = (self.description.iloc[idx, 6], self.description.iloc[idx, 7])
        gaze = (self.description.iloc[idx, 8], self.description.iloc[idx, 9])

        source_image = cv2.imread(self.home_path + "videogaze_images/" + source_name[1:])
        target_image = cv2.imread(self.home_path + "videogaze_images/" + target_name[1:])
        source_image = cv2.resize(source_image, (400, 400))
        target_image = cv2.resize(target_image, (400, 400))

        if flip == 1:
            source_image = cv2.flip(source_image, 1)
            target_image = cv2.flip(target_image, 1)
        ground = gen_gauss(source_image, gaze)
#----face
        pic = source_image
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
        face = face_numpy
        #face = np.transpose(face, (2, 0, 1))
# ----end face
        source_image = torch.FloatTensor(source_image).view(3, 400, 400)
        target_image = torch.FloatTensor(target_image).view(3, 400, 400)
        head_image = torch.FloatTensor(face).view(3, 400, 400)
        ground       = torch.FloatTensor(ground).view(1, 400, 400)

        if 1 >= gaze[0] and gaze[0] >= 0 and 1 >= gaze[1] and gaze[1] >= 0:
            is_looked = 1
        else:
            is_looked = 0


        return source_image, target_image, head_image, is_looked, ground, (eyes), (gaze)