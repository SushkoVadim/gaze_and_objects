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

root_path = "/local_scratch/vsushko/AAnaconda/02-25 transfer from caffe/"
path_test = root_path+"test"
data_descr_path = root_path+"data_descript"
path_train = root_path+"train"
model_path = root_path+"model"

net = convNet()
net.load_state_dict(torch.load("/local_scratch/vsushko/pycharm_proj/heatmaps/model.ptch"))

# DATA PREPARATION

