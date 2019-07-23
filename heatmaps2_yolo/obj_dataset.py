from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F

class objDataset(Dataset):
    def __init__(self, descr_file):
        self.description = pd.read_csv(descr_file, index_col=False)

    def __len__(self):
        return len(self.description)

    def __getitem__(self, idx):

        image = torch.load("/local_scratch/vsushko/heatmaps_compare/torch_data_trainannotations/torch_data_trainannotations/images_torch1/"+"{:03d}".format(int(idx/1000))+"/image_torch"+"{:06d}".format(idx))[0]
        obj = torch.load("/local_scratch/vsushko/heatmaps_compare/torch_data_trainannotations/torch_data_trainannotations/obj_torch1/"+"{:03d}".format(int(idx/1000))+"/obj_torch"+"{:06d}".format(idx))[0]
        ground = torch.load("/local_scratch/vsushko/heatmaps_compare/torch_data_trainannotations/torch_data_trainannotations/grounds_torch_eyes1/"+"{:03d}".format(int(idx/1000))+"/ground_torch_eyes" + "{:06d}".format(idx))[0]
        image_flip1 = torch.flip(image, dims=[1])
        obj_flip1 = torch.flip(obj, dims=[1])
        ground_flip1 = torch.flip(ground, dims=[0])
        image_flip2 = torch.flip(image, dims=[2])
        obj_flip2 = torch.flip(obj, dims=[2])
        ground_flip2 = torch.flip(ground, dims=[1])
        image_flip12 = torch.flip(image, dims=(1, 2))
        obj_flip12 = torch.flip(obj, dims=(1, 2))
        ground_flip12 = torch.flip(ground, dims=(0, 1))


        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        obj = obj.view(1, obj.shape[0], obj.shape[1], obj.shape[2])
        ground = ground.view(1, 1, ground.shape[0], ground.shape[1])
        image_flip1 = image_flip1.view(1, image_flip1.shape[0], image_flip1.shape[1], image_flip1.shape[2])
        obj_flip1 = obj_flip1.view(1, obj_flip1.shape[0], obj_flip1.shape[1], obj_flip1.shape[2])
        ground_flip1 = ground_flip1.view(1, 1, ground_flip1.shape[0], ground_flip1.shape[1])
        image_flip2 = image_flip2.view(1, image_flip2.shape[0], image_flip2.shape[1], image_flip2.shape[2])
        obj_flip2 = obj_flip2.view(1, obj_flip2.shape[0], obj_flip2.shape[1], obj_flip2.shape[2])
        ground_flip2 = ground_flip2.view(1, 1, ground_flip2.shape[0], ground_flip2.shape[1])
        image_flip12 = image_flip12.view(1, image_flip12.shape[0], image_flip12.shape[1], image_flip12.shape[2])
        obj_flip12 = obj_flip12.view(1, obj_flip12.shape[0], obj_flip12.shape[1], obj_flip12.shape[2])
        ground_flip12 = ground_flip12.view(1, 1, ground_flip12.shape[0], ground_flip12.shape[1])

        max_dim = max(image.shape[2], image.shape[3])
        min_dim = min(image.shape[2], image.shape[3])
        if max_dim > 1600:
            dim0 = int(image.shape[2] / (max_dim / 1600))
            dim1 = int(image.shape[3] / (max_dim / 1600))
            image = F.interpolate(image, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj = F.interpolate(obj, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground = F.interpolate(ground, size=(dim0, dim1), mode='bilinear', align_corners=True)
            image_flip1 = F.interpolate(image_flip1, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj_flip1 = F.interpolate(obj_flip1, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_flip1 = F.interpolate(ground_flip1, size=(dim0, dim1), mode='bilinear', align_corners=True)
            image_flip2 = F.interpolate(image_flip2, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj_flip2 = F.interpolate(obj_flip2, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_flip2 = F.interpolate(ground_flip2, size=(dim0, dim1), mode='bilinear', align_corners=True)
            image_flip12 = F.interpolate(image_flip12, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj_flip12 = F.interpolate(obj_flip12, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_flip12 = F.interpolate(ground_flip12, size=(dim0, dim1), mode='bilinear', align_corners=True)
        if min_dim < 250:
            dim0 = int(image.shape[2] / (min_dim / 250))
            dim1 = int(image.shape[3] / (min_dim / 250))
            image = F.interpolate(image, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj = F.interpolate(obj, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground = F.interpolate(ground, size=(dim0, dim1), mode='bilinear', align_corners=True)
            image_flip1 = F.interpolate(image_flip1, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj_flip1 = F.interpolate(obj_flip1, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_flip1 = F.interpolate(ground_flip1, size=(dim0, dim1), mode='bilinear', align_corners=True)
            image_flip2 = F.interpolate(image_flip2, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj_flip2 = F.interpolate(obj_flip2, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_flip2 = F.interpolate(ground_flip2, size=(dim0, dim1), mode='bilinear', align_corners=True)
            image_flip12 = F.interpolate(image_flip12, size=(dim0, dim1), mode='bilinear', align_corners=True)
            obj_flip12 = F.interpolate(obj_flip12, size=(dim0, dim1), mode='bilinear', align_corners=True)
            ground_flip12 = F.interpolate(ground_flip12, size=(dim0, dim1), mode='bilinear', align_corners=True)

        return [image, obj, ground,   image_flip1, obj_flip1, ground_flip1,
                image_flip2, obj_flip2, ground_flip2,  image_flip12, obj_flip12, ground_flip12]
