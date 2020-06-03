import sys
import os
import numpy as np
import cv2

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main():
    img_trainning_numbers = cv2.imread("training_chars.png")

    if img_trainning_numbers is None:
        print("error: img not exit\n")
        os.system("pause")
        return
    # end if

    img_gray = cv2.cvtColor(
        img_trainning_numbers, cv2.COLOR_BGR2GRAY)  # get gray img
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # filter img to binary form
    img_thresh = cv2.adaptiveThreshold(img_blurred,
                                       255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       11,
                                       2)

    cv2.imshow("imgThresh", img_thresh)

    img_thresh_cp = img_thresh.copy()
    npa_contours, _ = cv2.findContours(img_thresh_cp,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    # declare empty numpy array, it will store img data
    flattened_images = np.empty(
        (0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    # declare empty array to store input
    int_classifications = []
    int_chars = [ord('0'), ord('1'), ord('2'), ord('3'),
                 ord('4'), ord('5'), ord('6'), ord('7'),
                 ord('8'), ord('9'),
                 ord('A'), ord('B'), ord('C'), ord('D'),
                 ord('E'), ord('F'), ord('G'), ord('H'),
                 ord('I'), ord('J'), ord('K'), ord('L'),
                 ord('M'), ord('N'), ord('O'), ord('P'),
                 ord('Q'), ord('R'), ord('S'), ord('T'),
                 ord('U'), ord('V'), ord('W'), ord('X'),
                 ord('Y'), ord('Z')]

    for npa_contour in npa_contours:
        # if contour is big enough to consider
        if cv2.contourArea(npa_contour) > MIN_CONTOUR_AREA:
            [int_x, int_y, int_w, int_h] = cv2.boundingRect(
                npa_contour)         # get and break out bounding rect

            # draw rectangle around each contour as we ask user for input
            cv2.rectangle(img_trainning_numbers,        # draw rectangle
                          (int_x, int_y),               # upper left corner
                          (int_x+int_w, int_y+int_h),   # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            # crop char out of threshold image
            img_roi = img_thresh[int_y:int_y+int_h, int_x:int_x+int_w]
            # resize img, this will be more consistent for recog and storage
            img_roi_resized = cv2.resize(
                img_roi, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            # show resized image for reference
            cv2.imshow("imgROIResized", img_roi_resized)
            # show training, this will now have red rectangles drawn on it
            cv2.imshow("training_numbers.png", img_trainning_numbers)

            int_char = cv2.waitKey(0)                     # get key press
            if int_char == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            # else if the char is in the list of chars we are looking for . . .
            elif int_char in int_chars:

                # append classification char to integer list of chars
                # (we will convert to float later before writing to file)
                int_classifications.append(int_char)

                # flatten image to 1d numpy array so we can write to file later
                flattened_image = img_roi_resized.reshape(
                    (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                # add current flattened impage numpy array
                # to list of flattened image numpy arrays
                flattened_images = np.append(
                    flattened_images, flattened_image, 0)
            # end if
        # end if
    # end for

    flt_classifications = np.array(int_classifications, np.float32)
    classifications = flt_classifications.reshape(
        (flt_classifications.size, 1))

    print("\ntraining complete~\n")

    np.savetxt("classifications.txt", classifications)
    np.savetxt("flattened_images.txt", flattened_images)

    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
# end if
