import torchvision
from torchvision import transforms
import os
from PIL import Image

class selfData:
    def __init__(self, img_path, target_path, transforms = None):
        with open(target_path, 'r') as f:
	    lines = f.readlines()
	    self.img_list = [os.path.join(img_path, i.split()[0]) for i in lines]
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
