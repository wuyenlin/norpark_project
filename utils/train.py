from utils.dataloader import selfData, collate_fn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

def train(epoch, img_path, target_path, transforms, net, criterion):
    train_dataset = selfData(img_path, target_path, transforms)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 0,drop_last= False, collate_fn=collate_fn)
    for ep in range(epoch):  
        learning_rate = 0.01
        if ep >= 12:
            learning_rate = 0.0025
        elif ep >= 6:
            learning_rate = 0.005
        running_loss = 0.0
        print("Epoch {}.".format(ep+1))
        for i, data in enumerate(train_loader,1):
            inputs, labels = data
            labels = list(map(int, labels))
            labels = torch.Tensor(labels)
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                inputs = inputs.to(device)
                labels = labels.to(device)
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("Epoch {}.\tBatch {}.\tLoss = {:.3f}.".format(ep+1, i, running_loss))
    print('Finished Training.')