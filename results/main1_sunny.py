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

    args.train_img = 'CNRPark-EXT/PATCHES'
    args.train_lab = 'splits/CNRPark-EXT/sunny.txt'
    args.test_img = 'CNRPark-EXT/PATCHES/'
    args.test_lab = ['splits/CNRPark-EXT/overcast.txt','splits/CNRPark-EXT/rainy.txt', 'splits/PKLot/val.txt']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    txt_file = open(args.train_lab.split('/')[-1], 'a')
    txt_file.write("Start training: {}\n".format(datetime.now()))

    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 

    '''
    #uncomment to have a view of what your training dataset looks like.
    train_dataset = selfData(img_path2, target_path, transforms)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 0, drop_last= False)
    imgs, labels = train_loader.__iter__().__next__()
    imshow(train_loader)
    '''

    net = mAlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    # train(args.epochs, args.train_img, args.train_lab, transforms, net, criterion)

    PATH = './sunny.pth'
    # torch.save(net.state_dict(), PATH)
    net.load_state_dict(torch.load(PATH))

    for set in args.test_lab:       
        txt_file.write("\nStart time of testing on {}: {}\n".format(set.split('.')[0], datetime.now()))
        accuracy = test(args.test_img, set, transforms, net)
        txt_file.write("The accuracy of training on '{}' and testing on '{}' is {:.3f}.\n".format(args.train_lab.split('.')[0], set.split('.')[0], accuracy))
        txt_file.write("End time of testing on {}: {}\n".format(set.split('.')[0], datetime.now()))


    txt_file.close()