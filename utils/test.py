from utils.dataloader import selfData
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

def test(img_path, target_path, transforms, net):
    print("\nTesting starts now...")
    test_dataset = selfData(img_path, target_path, transforms)
    test_loader = DataLoader(test_dataset, batch_size = 100, shuffle = True, num_workers = 0)
    correct = 0
    total = 0
    item = 1
    with torch.no_grad():
        for i in range(len(test_loader)):
            try:
                data = test_loader[i]
            except FileNotFoundError:
                continue
            print("Testing on image {}".format(item))
            images, labels = data
            labels = list(map(int, labels))
            labels = torch.Tensor(labels)
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                images = images.to(device)
                labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            item += 1

    return (correct/total)