import cv2
import possibleChars
import preprocess
import numpy as np

kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


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


def check_if_possible_char(possibleChar):
    if (possibleChar.int_rect_area > MIN_PIXEL_AREA
        and possibleChar.int_rect_w > MIN_PIXEL_WIDTH
        and possibleChar.int_rect_h > MIN_PIXEL_HEIGHT
        and MIN_ASPECT_RATIO < possibleChar.flt_aspect_ratio
            and possibleChar.flt_aspect_ratio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def recognize_char(img_thresh, list_of_possible_chars):
    strChars = ""
    height, width = img_thresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    list_of_possible_chars.sort(
        key=lambda matchingChar: matchingChar.int_center_x)

    cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR, imgThreshColor)

    for currentChar in list_of_possible_chars:
        pt1 = (currentChar.int_rect_x, currentChar.int_rect_y)
        pt2 = ((currentChar.int_rect_x + currentChar.int_rect_w),
               (currentChar.int_rect_y + currentChar.int_rect_h))

        cv2.rectangle(imgThreshColor, pt1, pt2, (0.0, 255.0, 0.0),
                      2)           # draw green box around the char

        # crop char out of threshold image
        imgROI = img_thresh[currentChar.int_rect_y: currentChar.int_rect_y
                            + currentChar.int_rect_h,
                            currentChar.int_rect_x: currentChar.int_rect_x
                            + currentChar.int_rect_w]

        # resize image, this is necessary for char recognition
        imgROIResized = cv2.resize(
            imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))

        # flatten image into 1d numpy array
        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))

        # convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(
            npaROIResized, k=1)

        # get character from results
        strCurrentChar = str(chr(int(npaResults[0][0])))

        # append current char to full string
        strChars = strChars + strCurrentChar

    # end for
    return strChars


load_and_train_KNN()

img = cv2.imread("test.jpg")
img_gray, img_thresh = preprocess.preprocess(img)

list_of_possible_chars = []
img_thresh_copy = img_thresh.copy()

cv2.imshow('threshold', img_thresh)
cv2.waitKey(0)
contours, _ = cv2.findContours(
    img_thresh_copy, cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    possibleChar = possibleChars.PossibleChar(contour)
    if check_if_possible_char(possibleChar):
        list_of_possible_chars.append(possibleChar)

strChars = recognize_char(img_thresh, list_of_possible_chars)
print(strChars)
