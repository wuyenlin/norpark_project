#!/usr/bin/python3
from utils.options import args_parser
from utils.imshow import imshow
from model.malexnet import mAlexNet
from utils.dataloader import selfData
from utils.train_weather import train
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 

    args.epochs = 6
    net = mAlexNet().to(device)
    criterion = nn.CrossEntropyLoss()

    args.test_img = 'NORPark/'
    args.test_lab = 'splits/NORPark/nor_lab.txt'

    txt_file = open('nor_val.txt', 'a')
    txt_file.write("Start training: {}\n".format(datetime.now()))

    PATHS = ['trained_model/sunny.pth','trained_model/overcast.pth',
             'trained_model/rainy.pth','trained_model/04.pth',
             'trained_model/05.pth','trained_model/puc.pth']
    for PATH in PATHS:
        net = mAlexNet().to(device)
        net.load_state_dict(torch.load(PATH))
        accuracy = test(args.test_img, args.test_lab, transforms, net)
        trained_name = PATH.split('/')[-1].split('.jpg')[0]
        txt_file.write("'{}':\t{:.3f}.\n".format(trained_name, accuracy))
        print('\nAccuracy : {}'.format(accuracy))
    txt_file.close()