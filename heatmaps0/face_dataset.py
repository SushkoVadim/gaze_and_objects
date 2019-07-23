from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import torch

def gen_gauss(pic, gaze):
    x_grid, y_grid = np.meshgrid(np.linspace(1, pic.shape[1], pic.shape[1]), np.linspace(1,pic.shape[0], pic.shape[0]))
    sigma = 20
    mu = np.array([gaze[1]*pic.shape[1], gaze[0]*pic.shape[0]])
    d = np.sqrt((x_grid-mu[0])*(x_grid-mu[0])+(y_grid-mu[1])*(y_grid-mu[1]))
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    #g = 1*(d < 20)
    g = g/np.max(g)
    return g

class picDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, descr_file, images_root, face_path, image_mean, image_mean1):

        self.description = pd.read_csv(descr_file, index_col=False)
        self.images_root = images_root
        self.face_path = face_path
        self.image_mean = image_mean
        self.image_mean1 = image_mean1

    def __len__(self):
        return len(self.description)

    def __getitem__(self, idx):

        img_path = self.description.iloc[idx, 0]
        gaze = (self.description.iloc[idx, 8], self.description.iloc[idx, 9])

        image = cv2.imread(self.images_root+img_path)
        face = cv2.imread(self.face_path+'face'+str(idx+1)+'.jpg')

        max_dim = max(image.shape[0], image.shape[1])
        if max_dim > 1600:
            dim0 = int(image.shape[1] / (max_dim / 1600))
            dim1 = int(image.shape[0] / (max_dim / 1600))
            image = cv2.resize(image, (dim0, dim1))
            face = cv2.resize(face, (dim0, dim1))

        ground_truth = gen_gauss(image, gaze)

        image_mean = cv2.resize(self.image_mean, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        image = image - image_mean
        image = np.transpose(image, (2, 0, 1))

        image_mean1 = cv2.resize(self.image_mean1, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_CUBIC)
        face = face - image_mean1
        face = np.transpose(face, (2, 0, 1))

        sample = { 'image': torch.tensor(image).view(3, image.shape[1], image.shape[2]).float(),
                    'face': torch.tensor(face).view(3, image.shape[1], image.shape[2]).float(),
                  'ground': torch.tensor(ground_truth).view(1, image.shape[1], image.shape[2]).float()}

        return sample
