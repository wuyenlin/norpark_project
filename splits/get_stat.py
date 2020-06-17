#!/usr/bin/python3
def get_stat(file_name):
    f = open(file_name,'r')
    lines = f.readlines()
    occ, emp = 0, 0
    for i in lines:
        if i.split()[-1] == '1':
            occ += 1
        else:
            emp += 1
    print(file_name.split('/')[0])
    print("Occupied: {}.\nEmpty: {}.".format(occ, emp))
    print("Occ to emp ratio: {:.3f}\n".format(occ/emp))


if __name__ == "__main__":
    get_stat('CNRPark-EXT/all.txt')
    get_stat('PKLot/all.txt')
    get_stat('NORPark/nor_lab.txt')