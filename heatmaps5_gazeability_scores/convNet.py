import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from my_resnet18 import resnet18
import matplotlib.pyplot as plt


class scoreNet(nn.Module):
    def __init__(self):
        super(scoreNet, self).__init__()

        self.part_face = sumNet()
        self.part_face.load_state_dict(torch.load("data_descript/gaze_trained_model_after_0epoch.ptch"))

        self.part_obj = sumNet()
        self.part_obj.load_state_dict(torch.load("data_descript/flip_eyes_trained_model_after_0epoch.ptch"))

        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        self.fc1.weight.data = self.fc1.weight.data/100
        self.fc2.weight.data = self.fc2.weight.data/100
        self.fc3.weight.data = self.fc3.weight.data/100

    def forward(self, full, face, obj, gr_gaze, gr_eyes):
        out1 = self.part_face(full, face)[0]
        out1 = out1*gr_gaze
        out1 = torch.sum(out1)

        out2 = self.part_obj(full, obj)[0]
        out2 = out2*gr_eyes
        out2 = torch.sum(out2)

        out3 = torch.tensor([out1, out2]).to("cuda:0")

        final_out = self.fc1( out3 )
        final_out = F.relu(final_out)
        final_out = self.fc2(final_out)
        final_out = F.relu(final_out)
        final_out = self.fc3(final_out)
        final_out = F.softmax(final_out, dim=0).view(1, 2)

        return final_out


class sumNet(nn.Module):
    def __init__(self):
        super(sumNet, self).__init__()

        self.resnet1 = resnet18(pretrained=False)
        self.resnet1.fc = nn.Sequential(nn.Conv2d(512, 64, kernel_size=5, padding=2),
                                       nn.Conv2d(64, 32, kernel_size=5, padding=2),
                                       nn.Conv2d(32, 1, kernel_size=5, padding=2))
        self.resnet1.load_state_dict(torch.load("data_descript/resnet_places_with_random_last.ptch"))

        self.resnet2 = resnet18(pretrained=False)
        self.resnet2.fc = nn.Sequential(nn.Conv2d(512, 64, kernel_size=5, padding=2),
                       nn.Conv2d(64, 32, kernel_size=5, padding=2),
                       nn.Conv2d(32, 1, kernel_size=5, padding=2))
        self.resnet2.load_state_dict(torch.load("data_descript/resnet_imagenet_with_random_last.ptch"))

    def forward(self, full, face):

        # --part full
        x = full
        x = self.resnet1(x)
        x = F.interpolate(x, size=(full.shape[2], full.shape[3]), mode='bilinear', align_corners=False)
        out_data = x

        # --part face
        x = face
        x = self.resnet2(x)
        x = F.interpolate(x, size=(face.shape[2], face.shape[3]), mode='bilinear', align_corners=False)
        out_eyes = x

        # --part final
        final_out = out_data * out_eyes
        final_out = F.sigmoid(final_out)

        return [final_out, out_data, out_eyes]
