from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


idx = 3

image = torch.load("torch_data/images_torch1/"+"{:03d}".format(int(idx/1000))+"/image_torch"+"{:06d}".format(idx))[0]
obj = torch.load("torch_data/obj_torch1/"+"{:03d}".format(int(idx/1000))+"/obj_torch"+"{:06d}".format(idx))[0]
ground = torch.load("torch_data/grounds_torch1/"+"{:03d}".format(int(idx/1000))+"/ground_torch" + "{:06d}".format(idx))[0]

image = np.array(image)
obj = np.array(obj)
ground = np.array(ground)

image = np.transpose(image, (1, 2, 0))
obj = np.transpose(obj, (1, 2, 0))
plt.imshow(image)
plt.show()
plt.imshow(obj)
plt.show()
plt.imshow(ground, cmap='hot')
plt.show()