# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
from vgg16 import VGG_16
import torch.nn.functional as F

if __name__ == "__main__":
    model = VGG_16()
    model.load_state_dict(torch.load("./pretrained/vgg_face_dag.pth"))
    model.eval()
    im = cv2.imread("./images/Aamir_Khan1.png")
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224)
    im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
    preds = F.softmax(model(im), -1)
    values, index = preds.max(-1)
    with open("./images/names.txt", 'r') as f:
        names = f.readlines()
    print("Index: %d, Confidence: %f, Name: %s" % (index, values, names[index]))
