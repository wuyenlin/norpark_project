# cv_project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1hBCAI1Px-e1sdegW64wwbdUqlfvbClXi/view?usp=sharing)


This final project is part of the requirements of the course CS4245 Seminar Computer Vision by Deep Learning (2019/2020) at TU Delft. The work is done by group 8, including Hao Liu, Sigurd Totland, and Yen-Lin Wu.

In this final project, we attempt to reproduce the result of [CNRPark+EXT](http://cnrpark.it/)- more specifically, table 2 and figure 5 in the paper, [Deep learning for decentralized parking lot occupancy detection](https://www.sciencedirect.com/science/article/abs/pii/S095741741630598X).

### Run
Clone the repository and download the image dataset. Run the code as follows:

> python3 [main.py](main.py)

By default, it runs `epochs=18`, train on `CNRPark Even` and test on `CNRPark Odd`.
The setting can be changed as shown in follows. For example, 

> python3 main.py --epochs 6 --train_img PKLot/PKLotSegmented/ --train_lab splits/PKLot/UFPR04.txt --test_img PKLot/PKLotSegmented/ --test_lab splits/PKLot/UFPR04_test.txt

If a trained model is to be loaded and test on other dataset ( i.e. `.pth` file exists), or AlexNet is to be used, run the following command:

> python3 main.py --path sunny.pth --model AlexNet

See arguments in [options.py](utils/option.py).

### Dataset
In this project, we not only used the PKLot and CNRPark dataset for training and testing, but also created our own dataset by fetching a camera image from Trondheim, Norway that is accessible online on [Inescam](https://www.insecam.org/). We made our own [image-segmentation toolbox](https://github.com/wuyenlin/image_segmentation) to segment and label each image.

### Requirements
```
python >= 3.6
pytorch >= 0.4
```
