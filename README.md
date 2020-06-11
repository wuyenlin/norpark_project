# cv_project
CV Project Repository

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IldHTJLOqBM0Ex8LOQKyEQrqWBVGslZw?authuser=1)


This final project is part of the requirement of the course CS4245 Seminar Computer Vision by Deep Learning (2019/2020) at TU Delft. The work is done by group 8, including Hao Liu, Sigurd Totland, and Yen-Lin Wu.

In this final project, we attempt to reproduce the result of ![CNRPark+EXT](http://cnrpark.it/).

### Run
Clone the repository and download the image dataset. Run the code as follows:
```
python3 main.py
```
By default, it runs `epochs=18`, train on `CNRPark Even` and test on `CNRPark Odd`.
The setting can be changed as shown in follows. See arguments in options.py. For example, 
```
python3 main.py --epochs 6 --train_img PKLot/PKLotSegmented/ --train_lab splits/PKLot/UFPR04.txt --test_img PKLot/PKLotSegmented/ --test_lab splits/PKLot/UFPR04_test.txt

```

### Requirements
```
python >= 3.6
pytorch >= 0.4
```
