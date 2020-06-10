#!/usr/bin/python3
from utils.imshow import imshow
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from utils.dataloader import selfData
from utils.train import train

img_path = 'PKLot/PKLotSegmented'
img_path2 = 'CNRPark-Patches-150x150/'
target_path2 = 'splits/PKLot/PUC_test.txt'
target_path3 = 'splits/PKLot/UFPR04_test.txt'
target_path4 = 'splits/PKLot/UFPR05_test.txt'
target_path5 = 'splits/CNRParkAB/odd.txt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),  # normalize to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) 

target_path = 'splits/CNRParkAB/even.txt'
img_path = 'CNRPark-Patches-150x150'
train_dataset = selfData(img_path, target_path, transforms)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 0,drop_last= False)
imgs, labels = train_loader.__iter__().__next__()
imshow(train_loader)

net = mAlexNet().to(device)

criterion = nn.CrossEntropyLoss()

train(train_loader, net)
PATH = './cifar_net_even.pth'
torch.save(net.state_dict(), PATH)
net.load_state_dict(torch.load(PATH))

test(img_path2, target_path5, transforms)