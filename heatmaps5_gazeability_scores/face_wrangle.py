import sys
import torch
import pdb
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

sys.path.insert(0, '/local_scratch/vsushko/face_recognition/FaceDetection_DSFD')
from data import *
from face_ssd import build_ssd

description = pd.read_csv('/local_scratch/vsushko/face_recognition/train_annotations.txt', header=None)
root_path = '/local_scratch/vsushko/face_recognition/'
out_file = pd.DataFrame()


torch.set_grad_enabled(False)
cfg = widerface_640
num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
net.load_state_dict(torch.load("FaceDetection_DSFD/weights/WIDERFace_DSFD_RES152.pth", map_location='cpu'))
net.cuda()
net.eval()
transform = TestBaseTransform((104, 117, 123))
thresh=cfg['conf_thresh']
cuda = True
visual_threshold = 0.3

if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# load net
print('Finished loading model!')

# evaluation

transform = TestBaseTransform((104, 117, 123))
thresh = cfg['conf_thresh']
# save_path = args.save_folder
# num_images = len(testset)

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        try:
            max_score = np.max(det_accu[:, 4]).item()
        except:
            max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))

        try:
            det_accu_sum[:, 0:4] = np.array(np.sum(det_accu[:, 0:4], axis=0)) / np.array(np.sum(det_accu[:, -1:]).cpu())
        except:
            det_accu_sum[:, 0:4] = np.array(np.sum(det_accu[:, 0:4], axis=0)) / np.array(np.sum(det_accu[:, -1:]))
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets


def write_to_txt(f, det, event, im_name):
    f.write('{:s}\n'.format(event + '/' + im_name))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))


def infer(net, img, transform, thresh, cuda, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    x = x.cuda()
    # print (shrink , x.shape)
    y = net(x)  # forward pass

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1] / shrink, img.shape[0] / shrink,
                          img.shape[1] / shrink, img.shape[0] / shrink])
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            # label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            det.append([pt[0], pt[1], pt[2], pt[3], score])
            j += 1
    if (len(det)) == 0:
        det = [[0.1, 0.1, 0.2, 0.2, 0.01]]
    det = np.array(det)

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def infer_flip(net, img, transform, thresh, cuda, shrink):
    img = cv2.flip(img, 1)
    det = infer(net, img, transform, thresh, cuda, shrink)
    det_t = np.zeros(det.shape)
    det_t[:, 0] = img.shape[1] - det[:, 2]
    det_t[:, 1] = det[:, 1]
    det_t[:, 2] = img.shape[1] - det[:, 0]
    det_t[:, 3] = det[:, 3]
    det_t[:, 4] = det[:, 4]
    return det_t


def infer_multi_scale_sfd(net, img, transform, thresh, cuda, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = infer(net, img, transform, thresh, cuda, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = infer(net, img, transform, thresh, cuda, bt)
    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, infer(net, img, transform, thresh, cuda, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, infer(net, img, transform, thresh, cuda, max_im_shrink)))
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b




# load data
for idx in range(0, len(description)):
    img_path = description.iloc[idx, 0]
    img_id = 'face'
    img = cv2.imread(root_path+img_path, cv2.IMREAD_COLOR)

    max_im_shrink = ((2000.0 * 2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    det0 = infer(net, img, transform, thresh, cuda, shrink)
    det1 = infer_flip(net, img, transform, thresh, cuda, shrink)
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = infer(net, img, transform, thresh, cuda, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    factor = 2
    bt = min(factor, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = infer(net, img, transform, thresh, cuda, bt)
    # enlarge small iamge x times for small face
    #if max_im_shrink > factor:
    #    bt *= factor
    #    while bt < max_im_shrink:
    #        det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , bt)))
    #        bt *= factor
    #    det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , max_im_shrink) ))
    #enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    det = np.row_stack((det0, det1, det_s, det_b))
    det = bbox_vote(det)

    class_name = 'face'
    dets = det
    inds = np.where(dets[:, -1] >= visual_threshold)[0]

    # print(len(inds))
    bbx_list = list()
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        bbx_list.append(bbox)
    out_file = out_file.append(pd.DataFrame({"idx": [idx], "len": [len(inds)],
                                         "bb": [(bbx_list)]}), ignore_index=True)
    if idx % 1 == 0:
        print(idx, len(inds), img.shape)

    if idx % 10 == 9:
        out_file.to_csv("faces"+str(idx)+".txt")
        out_file = pd.DataFrame()

out_file.to_csv("faces.txt")


