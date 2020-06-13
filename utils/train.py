from utils.dataloader import selfData

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

def train(epoch, img_path, target_path, transforms, net, criterion):
    train_dataset = selfData(img_path, target_path, transforms)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 0,drop_last= False)
    for ep in range(epoch):  
        if ep is 0:
            learning_rate = 0.01
        elif ep is 6:
            learning_rate = 0.005
        elif ep is 12:
            learning_rate = 0.0025
        running_loss = 0.0
        print("Epoch {}.".format(ep+1))
        for i, data in enumerate(train_loader,1):
            try:
                inputs, labels = data
            except FileNotFoundError:
                continue
            labels = list(map(int, labels))
            labels = torch.Tensor(labels)
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                inputs = inputs.to(device)
                labels = labels.to(device)
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("Epoch {}.\tImage {}.\tLoss = {:.3f}.".format(ep+1, i+1, running_loss))
            if i % 2000 == 1999:    # 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training.')