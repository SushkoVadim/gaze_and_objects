import pandas as pd
import numpy as np
import cv2
import torch
import scipy.io as sio
import os

for i in range(9915, 9925):
    a = torch.load("grounds_torch1/009/ground_torch00"+str(i))
    print(i, a[0][0, 0])