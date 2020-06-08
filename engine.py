import cv2
import numpy as np
import os
import preprocess
import argparse
from sklearn.svm import SVC

import possibleChars

find_char = False
get_accuracy = False

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="Path to the plate image")
ap.add_argument("-f", "--folder", default=None,
                help="Path to the folder plate images")
ap.add_argument("-m", "--model",
                required=True)
args = vars(ap.parse_args())
if args['image'] is not None:
    find_char = True
    if not os.path.exists(args['image']):
        print("The image path doesn't exist!")
        find_char = False
if args['folder'] is not None:
    get_accuracy = True
    if not os.path.exists(args['folder']):
        print("The folder path doesn't exist!")
        get_accuracy = False
args = vars(ap.parse_args())

if(args["model"] == 'SVM'):
    svclassifier = SVC(kernel='linear')
elif(args["model"] == 'KNN'):
    kNearest = cv2.ml.KNearest_create()


MIN_PIXEL_WIDTH = 1
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 0.9

MIN_PIXEL_AREA = 63

MAX_CHANGE_IN_AREA = 0.5

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 15


def load_and_train_KNN():
    try:
        # read in training classifications
        classifications = np.loadtxt("classifications.txt", np.float32)
    # if file could not be opened
    except Exception:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        # and return False
        return False

    try:
        # read in training images
        flattened_images = np.loadtxt("flattened_images.txt", np.float32)
    # if file could not be opened
    except Exception:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return False

    # reshape numpy array to 1d, necessary to pass to call to train
    classifications = classifications.reshape(
        (classifications.size, 1))

    # set default K to 1
    kNearest.setDefaultK(1)

    kNearest.train(flattened_images, cv2.ml.ROW_SAMPLE,
                   classifications)           # train KNN object

    # if we got here training was successful so return true
    return True
# end function


def load_and_train_SVM():
    try:
        # read in training classifications
        classifications = np.loadtxt("classifications.txt", np.float32)
    # if file could not be opened
    except Exception:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        # and return False
        return False

    try:
        # read in training images
        flattened_images = np.loadtxt("flattened_images.txt", np.float32)
    # if file could not be opened
    except Exception:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return False

    # reshape numpy array to 1d, necessary to pass to call to train
    classifications = classifications.ravel()

    svclassifier.fit(flattened_images, classifications)
    return True


def find_chars(img_thresh):
    list_of_chars = []
    img_thresh_cp = img_thresh.copy()
    cv2.rectangle(img_thresh_cp, (0, 0), (
        img_thresh.shape[1],
        img_thresh.shape[0]),
        (0, 0, 0), 3)
    contours, _ = cv2.findContours(
        img_thresh_cp, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        possibleChar = possibleChars.PossibleChar(contour)
        if check_if_char(possibleChar):
            list_of_chars.append(possibleChar)
    return list_of_chars


def check_if_char(possibleChar):
    if (possibleChar.int_rect_area > MIN_PIXEL_AREA
        and possibleChar.int_rect_w > MIN_PIXEL_WIDTH
        and possibleChar.int_rect_h > MIN_PIXEL_HEIGHT
        and MIN_ASPECT_RATIO < possibleChar.flt_aspect_ratio
            and possibleChar.flt_aspect_ratio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def recognize_char(img_thresh, list_of_chars):
    str_chars = ""
    height, width = img_thresh.shape

    img_thresh_color = np.zeros((height, width, 3), np.uint8)

    list_of_chars.sort(
        key=lambda matching_char: matching_char.int_center_x)

    cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR, img_thresh_color)

    for current_char in list_of_chars:
        pt1 = (current_char.int_rect_x, current_char.int_rect_y)
        pt2 = ((current_char.int_rect_x + current_char.int_rect_w),
               (current_char.int_rect_y + current_char.int_rect_h))
        # crop char out of threshold image
        imgROI = img_thresh[current_char.int_rect_y: current_char.int_rect_y
                            + current_char.int_rect_h,
                            current_char.int_rect_x: current_char.int_rect_x
                            + current_char.int_rect_w]

        # resize image, this is necessary for char recognition
        imgROIResized = cv2.resize(
            imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))

        # flatten image into 1d numpy array
        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))

        # convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)
        # SVM
        if(args["model"] == 'SVM'):
            npaResults = svclassifier.predict(npaROIResized)
            str_current_char = chr(int(npaResults[0]))
        elif(args["model"] == 'KNN'):
            # Kneares
            retval, npaResults, neigh_resp, dists = kNearest.findNearest(
                npaROIResized, k=1)

            # get character from results
            str_current_char = str(chr(int(npaResults[0][0])))

            # append current char to full string
        str_chars = str_chars + str_current_char

    # end for
    return str_chars


def main():
    if(args["model"] == 'SVM'):
        load_and_train_SVM()
    elif(args["model"] == 'KNN'):
        load_and_train_KNN()
    if find_char:
        img = cv2.imread(args['image'])
        img_gray, img_thresh = preprocess.preprocess(img)
        list_of_chars = find_chars(img_thresh)
        str_plate = recognize_char(img_thresh, list_of_chars)
        print(str_plate)
    elif get_accuracy:
        count_images = 0
        count_true_plate = 0
        count_chars = 0
        count_true = 0
        for image in os.listdir(args['folder']):
            count_images += 1
            label = image[9:17]
            pathImage = args['folder'] + "/" + image
            img = cv2.imread(pathImage)
            img_gray, img_thresh = preprocess.preprocess(img)
            list_of_chars = find_chars(img_thresh)
            str_plate = recognize_char(img_thresh, list_of_chars)
            if len(str_plate) < len(label):
                for i in range(0, len(label) - len(str_plate)):
                    str_plate = str_plate + '0'
            for i in range(0, len(label)):
                count_chars += 1
                if label[i] == str_plate[i]:
                    count_true += 1
            if (label == str_plate):
                count_true_plate += 1
            else:
                print(label)
                print(str_plate + "\n")
        print("Total chars: " + str(count_chars) + "\n")
        print("True chars: " + str(count_true) + "\n")
        print("Accuracy chars: " + str(float(count_true/count_chars)) + "\n")
        print("Accuracy plate: " + str(float(count_true_plate/count_images)))
    else:
        print("Nothing to do, pass args to do some thing!")
    return


if __name__ == '__main__':
    main()
