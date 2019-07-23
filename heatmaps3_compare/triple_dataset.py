from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F

class tripleDataset(Dataset):
    def __init__(self, descr_file):
        self.description = pd.read_csv(descr_file, index_col=False, header=None)

    def __len__(self):
        return len(self.description)

    def __getitem__(self, idx):

        image = torch.load("torch_data_trainannotations/torch_data_trainannotations/images_torch1/"+"{:03d}".format(int(idx/1000))+"/image_torch"+"{:06d}".format(idx))[0]
        obj = torch.load("torch_data_trainannotations/torch_data_trainannotations/obj_torch1/"+"{:03d}".format(int(idx/1000))+"/obj_torch"+"{:06d}".format(idx))[0]
        face = torch.load("torch_data_trainannotations/torch_data_trainannotations/face_torch1/" + "{:03d}".format(int(idx / 1000)) + "/face_torch" + "{:06d}".format(idx))[0]
        ground_gaze = torch.load("torch_data_trainannotations/torch_data_trainannotations/grounds_torch_gaze1/" + "{:03d}".format(int(idx / 1000)) + "/ground_torch_gaze" + "{:06d}".format(idx))[0]
        ground_eyes = torch.load("torch_data_trainannotations/torch_data_trainannotations/grounds_torch_eyes1/" + "{:03d}".format(int(idx / 1000)) + "/ground_torch_eyes" + "{:06d}".format(idx))[0]

        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        obj = obj.view(1, obj.shape[0], obj.shape[1], obj.shape[2])
        face = face.view(1, face.shape[0], face.shape[1], face.shape[2])
        ground_gaze = ground_gaze.view(1, 1, ground_gaze.shape[0], ground_gaze.shape[1])
        ground_eyes = ground_eyes.view(1, 1, ground_eyes.shape[0], ground_eyes.shape[1])

        max_dim = max(image.shape[2], image.shape[3])
        min_dim = min(image.shape[2], image.shape[3])
        if max_dim > 1600:
            dim0 = int(image.shape[2] / (max_dim / 1600))
            dim1 = int(image.shape[3] / (max_dim / 1600))
            image = F.interpolate(image, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj = F.interpolate(obj, size=(dim0, dim1), mode='bilinear', align_corners=True)
            face = F.interpolate(face, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_gaze = F.interpolate(ground_gaze, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_eyes = F.interpolate(ground_eyes, size=(dim0, dim1), mode='bilinear', align_corners=True)
        if min_dim < 250:
            dim0 = int(image.shape[2] / (min_dim / 250))
            dim1 = int(image.shape[3] / (min_dim / 250))
            image = F.interpolate(image, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj = F.interpolate(obj, size=(dim0, dim1), mode='bilinear', align_corners=True)
            face = F.interpolate(face, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_gaze = F.interpolate(ground_gaze, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_eyes = F.interpolate(ground_eyes, size=(dim0, dim1), mode='bilinear', align_corners=True)

        return [image, obj, face, ground_gaze, ground_eyes]