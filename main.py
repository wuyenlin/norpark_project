#!/usr/bin/python3
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

img_path = 'PKLot/PKLotSegmented'
img_path2 = 'CNRPark-Patches-150x150/'

target_path1 = 'splits/CNRParkAB/even.txt'
target_path2 = 'splits/PKLot/PUC_test.txt'
target_path3 = 'splits/PKLot/UFPR04_test.txt'
target_path4 = 'splits/PKLot/UFPR05_test.txt'
target_path5 = 'splits/CNRParkAB/odd.txt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### change here to run different experiments
epoch = 2
train_img_path = img_path2
train_lab_path = target_path1

test_img_path = img_path2
test_lab_path = target_path5
###

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

train(epoch, train_img_path, train_lab_path, transforms, net, criterion)
PATH = './cifar_net_even.pth'
torch.save(net.state_dict(), PATH)
net.load_state_dict(torch.load(PATH))
accuracy = test(test_img_path, test_lab_path, transforms, net)
print("\nThe accuracy of training on '{}' and testing on '{}' is {:.3f}.".format(train_lab_path.split('.')[0], test_lab_path.split('.')[0], accuracy))