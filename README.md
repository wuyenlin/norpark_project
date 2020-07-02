# cv_project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19WoAA0vinucKOj-dxMs8je-tKms8rZza?usp=sharing)


This final project is a part of the requirements of the course CS4245 Seminar Computer Vision by Deep Learning (2019/2020) at TU Delft. The work is done by group 8, including Hao Liu, Sigurd Totland, and Yen-Lin Wu.

In this final project, we attempted to reproduce the result of [CNRPark+EXT](http://cnrpark.it/)- more specifically, table 2 and figure 5 in the paper, [Deep learning for decentralized parking lot occupancy detection](https://www.sciencedirect.com/science/article/abs/pii/S095741741630598X).
We also created our own test set by downloading images of a surveillance camera in a parking lot at Trondheim, Norway (details can be found in the folder [NORPark](NORPark/)). We report both the reproduction results and the accuracy testing on NORPark. A full blog post can be read [here](CV Blog Post.pdf).

### Download Dataset
Run the following commands to download the CNRPark, CNRPark-EXT, and PKLot dataset used in this project.

> chmod +x get_dataset.sh

> ./get_dataset.sh

### Run files
Clone the repository and download the image dataset. Run the code as follows:

> python3 [main_tab2.py](main_tab2.py)

By default, it runs `epochs=18`, train on `CNRPark Even` and test on `CNRPark Odd`.
If a trained model is to be loaded and test on other dataset ( i.e. `.pth` file exists), or AlexNet is to be used, run the following command:

> python3 main_tab2.py --path sunny.pth --model AlexNet

See arguments in [options.py](utils/option.py).

### NORPark
We made a [segmentation and labeling toolbox](https://github.com/wuyenlin/image_segmentation) to process the obtained images.
Here is an example of the camera footage on the parking lot.
![](https://i.imgur.com/UBQGsgX.jpg)
The segmented images are zipped in [nor.zip](NORPark/PATCHES/nor.zip) and their corresponding label file is in 

### Requirements
```
python >= 3.6
pytorch >= 0.4
```
