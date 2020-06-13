import argparse

'''
    img_path1 = 'PKLot/PKLotSegmented'
    img_path2 = 'CNRPark-Patches-150x150/'

    target_path1 = 'splits/CNRParkAB/even.txt'
    target_path2 = 'splits/CNRParkAB/odd.txt'
    target_path3 = 'splits/PKLot/PUC_test.txt'
    target_path4 = 'splits/PKLot/UFPR04_test.txt'
    target_path5 = 'splits/PKLot/UFPR05_test.txt'
'''    

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=18, help="rounds of training")
    parser.add_argument('--imshow', type=bool, default=False, help="show some training dataset")
    # parser.add_argument('--model', type=str, default='mAlexNet', help='model name')
    parser.add_argument('--train_img', type=str, default='CNRPark-Patches-150x150/', help="path to training set images")
    parser.add_argument('--train_lab', type=str, default='splits/CNRParkAB/even.txt', help="path to training set labels")
    parser.add_argument('--test_img', type=str, default='CNRPark-Patches-150x150/', help="path to test set images")
    parser.add_argument('--test_lab', type=str, default='splits/CNRParkAB/odd.txt', help="path to test set labels")
    args = parser.parse_args()
    return args
