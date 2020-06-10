import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

def train(train_loader, net):
    for epoch in range(18):  
        if epoch is 0:
            learning_rate = 0.01
        elif epoch is 6:
            learning_rate = 0.005
        elif epoch is 12:
            learning_rate = 0.0025
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
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
            print("Round {}. Loss = {}.".format(i, running_loss))
            if i % 2000 == 1999:    # 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')