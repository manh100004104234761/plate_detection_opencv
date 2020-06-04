import random
import shutil
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
                help="Path to the folder plate image")
args = vars(ap.parse_args())
if not os.path.exists(args['folder']):
    print("The folder path doesn't exist!")
    exit()
args = vars(ap.parse_args())


def main(folder):
    datas = []
    for data in os.listdir(folder):
        datas.append(data)
    random.shuffle(datas)
    number_test = int(len(datas)/10)
    train_data = datas[:-number_test]
    test_data = datas[-number_test:]
    os.mkdir('train_'+folder)
    os.mkdir('test_'+folder)
    for i in train_data:
        shutil.copy(os.path.join(folder, i), 'train_'+folder)
    for i in test_data:
        shutil.copy(os.path.join(folder, i), 'test_'+folder)
    return


if __name__ == '__main__':
    main(args['folder'])
