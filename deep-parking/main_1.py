import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from malexnet_torch import mAlexnet

# normalize = transforms.Normalize(
#     mean = [0.485, 0.456, 0.406], 
#     std = [0.229, 0.224, 0.225])

'''
1. each figure in the dataset is around 30*60, but they are unique
2. In the main.py file, there is a sentence "scale = 1. /256;
I don`t know how to transfer it into pytorch" 
'''
transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224), 
#     transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
#     normalize  
]) 

class selfData(Dataset):
    def __init__(self, img_path, target_path, transforms = None):
        with open(target_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [
                os.path.join(img_path, i.split()[0]) for i in lines
            ]
            self.label_list = [i.split()[1] for i in lines]
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        img = self.transforms(img)
        label = self.label_list[index]

        return img, label
    
    def __len__(self):
        return len(self.label_list)

target_path = 'LABELS/train.txt'
img_path = 'PATCHES/'

train_dataset = selfData(img_path, target_path, transforms)
train_loader = DataLoader(train_dataset, batch_size = 100, shuffle = True, num_workers = 0)
img, label= train_loader.__iter__().__next__()

## show images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = mAlexnet()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

## training ##
for epoch in range(18):  # loop over the dataset 

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print("input: ",inputs.shape)
        labels = list(map(int, labels))
#         print(labels)
        labels = torch.Tensor(labels)
#         print(labels.shape)
#         inputs = inputs.view(-1,3)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs)
#         outputs.view(-1, 1)
        print("output : ", outputs.shape)
        print("labels : ", labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Training ended.')