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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    net = mAlexNet().to(device)
    criterion = nn.CrossEntropyLoss()

    args.train_img = 'CNRPark-EXT/PATCHES'
    args.train_lab = ['splits/CNRPark-EXT/sunny.txt','splits/CNRPark-EXT/overcast.txt','splits/CNRPark-EXT/rainy.txt']
    args.test_img = 'CNRPark-EXT/PATCHES/'
    args.test_lab = ['splits/CNRPark-EXT/sunny.txt','splits/CNRPark-EXT/overcast.txt','splits/CNRPark-EXT/rainy.txt','splits/PKLot/val.txt']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    txt_file = open('fig5_results.txt', 'a')
    txt_file.write("Start training: {}\n".format(datetime.now()))

    for train_set in args.train_lab:
        train(args.epochs, args.train_img, train_set, transforms, net, criterion)
        PATH = './trained.pth'
        torch.save(net.state_dict(), PATH)
        net.load_state_dict(torch.load(PATH))
        for test_set in args.test_lab:
            if test_set == train_set:
                accuracy = 'none'
                print("Skip to next test.")
            elif test_set == 'splits/PKLot/val.txt':
                args.test_img = 'PKLot/PKLotSegmented'
                accuracy = test(args.test_img, test_set, transforms, net)
            else:
                accuracy = test(args.test_img, test_set, transforms, net)
            print("Training on '{}' and testing on '{}': {:.3f}.\n".format(train_set.split('.')[0], test_set.split('.')[0], accuracy))
            txt_file.write("Training on '{}' and testing on '{}': {:.3f}.\n".format(train_set.split('.')[0], test_set.split('.')[0], accuracy))
    print("Experiments ended.")
    txt_file.close()