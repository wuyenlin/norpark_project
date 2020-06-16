#!/usr/bin/python3
from utils.options import args_parser
from utils.imshow import imshow
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from utils.dataloader import selfData
from utils.train import train
from utils.test import test

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

if __name__=="__main__":
    args = args_parser()

    args.test_img = './'
    args.test_lab = 'splits/no/nor_lab.txt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    txt_file = open('no_val.txt', 'a')
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 

    net = mAlexNet().to(device)
    criterion = nn.CrossEntropyLoss()

    PATHS = ['./sunny.pth','./overcast.pth','./rainy.pth','./04.pth','./05.pth','puc.pth']
    for PATH in PATHS:
        net.load_state_dict(torch.load(PATH))
        accuracy = test(args.test_img, args.test_lab, transforms, net)
        txt_file.write("'{}':\t{:.3f}.\n".format(PATH.split('.pth')[0], accuracy))
        print('\nAccuracy : {}'.format(accuracy))

    txt_file.close()