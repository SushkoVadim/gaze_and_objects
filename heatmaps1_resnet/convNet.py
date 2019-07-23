import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from my_resnet18 import resnet18
import matplotlib.pyplot as plt


class sumNet(nn.Module):
    def __init__(self):
        super(sumNet, self).__init__()

        self.convNet = convNet()
        self.convNet.load_state_dict(torch.load("model.ptch"))

        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Sequential(nn.Conv2d(512, 64, kernel_size=5, padding=2),
                       nn.Conv2d(64, 32, kernel_size=5, padding=2),
                       nn.Conv2d(32, 1, kernel_size=5, padding=2))
        self.resnet.load_state_dict(torch.load("resnet_with_random_last.ptch"))

    def forward(self, full, face):
        # --part full
        x = full
        x = self.convNet.conv1(x)
        x = self.convNet.relu(x)
        x = self.convNet.pool(x)
        x = self.convNet.conv2(x)
        x = self.convNet.relu(x)
        x = self.convNet.pool(x)
        x = self.convNet.conv3(x)
        x = self.convNet.relu(x)
        x = self.convNet.conv4(x)
        x = self.convNet.relu(x)
        x = self.convNet.conv5(x)
        x = self.convNet.relu(x)
        x = self.convNet.conv6(x)

        # x = self.relu(x)
        x = F.interpolate(x, size=(full.shape[2], full.shape[3]), mode='bilinear', align_corners=False)
        out_data = x

        #plt.imshow(out_data.detach().numpy()[0, 0, :, :], cmap='hot', interpolation='nearest')
        #plt.show()

        # --part face
        x = face
        x = self.resnet(x)

        x = F.interpolate(x, size=(face.shape[2], face.shape[3]), mode='bilinear', align_corners=False)
        out_eyes = x
        #plt.imshow(out_eyes.detach().numpy()[0, 0, :, :], cmap='hot', interpolation='nearest')
        #plt.show()

        # --part final
        final_out = out_data * out_eyes
        #plt.imshow(final_out.detach().numpy()[0, 0, :, :], cmap='hot', interpolation='nearest')
        #plt.show()
        final_out = F.sigmoid(final_out)

        return [final_out, out_data, out_eyes]

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=4)
        self.conv2 = nn.Conv2d(96, 256, (5, 5), groups=2, padding=2)
        self.conv3 = nn.Conv2d(256, 384, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, (3, 3), groups=2, padding=1)
        self.conv5 = nn.Conv2d(384, 256, (3, 3), groups=2, padding=1)
        self.conv6 = nn.Conv2d(256, 1, (1, 1))

        self.conv1face = nn.Conv2d(3, 30, (11, 11), stride=2)
        self.conv2face = nn.Conv2d(30, 60, (11, 11))
        self.conv3face = nn.Conv2d(60, 128, (9, 9))
        self.conv4face = nn.Conv2d(128, 256, (7, 7))
        self.conv5face = nn.Conv2d(256, 256, (7, 7))
        self.conv6face = nn.Conv2d(256, 128, (5, 5))
        self.conv7face = nn.Conv2d(128, 90, (3, 3))
        self.conv8face = nn.Conv2d(90, 48, (3, 3))
        self.conv9face = nn.Conv2d(48, 1, (3, 3))

        self.norm = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)
        self.pool = nn.MaxPool2d((3, 3), stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, full, face):
        # --part full
        x = full

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        # print(x)

        # x = self.relu(x)
        x = F.interpolate(x, size=(full.shape[2], full.shape[3]), mode='bilinear', align_corners=False)
        out_data = x

        # --part face
        x = face

        x = self.conv1face(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2face(x)
        x = self.relu(x)
        x = self.conv3face(x)
        x = self.relu(x)
        x = self.conv4face(x)
        x = self.relu(x)
        x = self.conv5face(x)
        x = self.relu(x)
        x = self.conv6face(x)
        x = self.relu(x)
        x = self.conv7face(x)
        x = self.relu(x)
        x = self.conv8face(x)

        # x = self.relu(x)

        x = F.interpolate(x, size=(face.shape[2], face.shape[3]), mode='bilinear', align_corners=False)
        out_eyes = x

        # --part final
        final_out = out_data * out_eyes
        final_out = F.sigmoid(final_out)

        return [final_out, out_eyes, out_data]