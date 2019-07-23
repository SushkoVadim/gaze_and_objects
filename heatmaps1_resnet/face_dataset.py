from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F

class picDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, descr_file, images_root, face_path):

        self.description = pd.read_csv(descr_file, index_col=False)
        self.images_root = images_root
        self.face_path = face_path

    def __len__(self):
        return len(self.description)

    def __getitem__(self, idx):

        image = torch.load("images_torch1/"+"{:03d}".format(int(idx/1000))+"/image_torch"+"{:06d}".format(idx))[0]
        face = torch.load("faces_torch1/"+"{:03d}".format(int(idx/1000))+"/face_torch"+"{:06d}".format(idx))[0]
        ground = torch.load("grounds_torch1/"+"{:03d}".format(int(idx/1000))+"/ground_torch" + "{:06d}".format(idx))[0]

        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        face = face.view(1, face.shape[0], face.shape[1], face.shape[2])
        ground = ground.view(1, 1, ground.shape[0], ground.shape[1])

        max_dim = max(image.shape[2], image.shape[3])
        min_dim = min(image.shape[2], image.shape[3])
        if max_dim > 1600:
            dim0 = int(image.shape[2] / (max_dim / 1600))
            dim1 = int(image.shape[3] / (max_dim / 1600))
            image = F.interpolate(image, size=(dim0, dim1), mode='bilinear', align_corners=True)
            face = F.interpolate(face, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground = F.interpolate(ground, size=(dim0, dim1), mode='bilinear', align_corners=True)
        if min_dim < 250:
            dim0 = int(image.shape[2] / (min_dim / 250))
            dim1 = int(image.shape[3] / (min_dim / 250))
            image = F.interpolate(image, size=(dim0, dim1), mode='bilinear', align_corners=True)
            face = F.interpolate(face, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground = F.interpolate(ground, size=(dim0, dim1), mode='bilinear', align_corners=True)

        return [image, face, ground]
