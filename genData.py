import sys
import os
import numpy as np
import possibleChars
import cv2

MIN_CONTOUR_AREA = 15

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

MIN_PIXEL_WIDTH = 1
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 0.9

MIN_PIXEL_AREA = 63


def check_if_char(possibleChar):
    if (possibleChar.int_rect_area > MIN_PIXEL_AREA
        and possibleChar.int_rect_w > MIN_PIXEL_WIDTH
        and possibleChar.int_rect_h > MIN_PIXEL_HEIGHT
        and MIN_ASPECT_RATIO < possibleChar.flt_aspect_ratio
            and possibleChar.flt_aspect_ratio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def main():
    # declare empty array to store input
    int_classifications = []
    # declare empty numpy array, it will store img data
    flattened_images = np.empty(
        (0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    for image in os.listdir('train_plate_car'):
        img_trainning_numbers = cv2.imread('train_plate_car/'+image)
        label = image[9:17]
        if img_trainning_numbers is None:
            print("error: img not exit\n")
            os.system("pause")
            return
        # end if

        img_gray = cv2.cvtColor(
            img_trainning_numbers, cv2.COLOR_BGR2GRAY)  # get gray img
        img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # filter img to binary form
        img_thresh = cv2.adaptiveThreshold(img_gray,
                                           255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           11,
                                           2)

        cv2.imshow("imgThresh", img_thresh)

        img_thresh_cp = img_thresh.copy()

        cv2.rectangle(img_thresh_cp, (0, 0), (
            img_trainning_numbers.shape[1],
            img_trainning_numbers.shape[0]),
            (0, 0, 0), 3)
        cv2.imshow("test", img_thresh_cp)
        npa_contours, _ = cv2.findContours(img_thresh_cp,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

        list_of_chars = []
        for npa_contour in npa_contours:
            possibleChar = possibleChars.PossibleChar(npa_contour)
            if check_if_char(possibleChar):
                list_of_chars.append(possibleChar)

        list_of_chars.sort(
            key=lambda matching_char: matching_char.int_center_x)

        for i in range(0, len(list_of_chars)):
            if i > 7:
                continue
            [int_x, int_y, int_w, int_h] = [list_of_chars[i].int_rect_x,
                                            list_of_chars[i].int_rect_y,
                                            list_of_chars[i].int_rect_w,
                                            list_of_chars[i].int_rect_h]
            img_roi = img_thresh[int_y:int_y+int_h, int_x:int_x+int_w]
            img_roi_resized = cv2.resize(
                img_roi, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            cv2.imshow("imgROIResized", img_roi_resized)
            save_file = os.path.join(
                "train_data", label[i], image + str(i) + ".jpg")
            cv2.imwrite(save_file, img_roi_resized)
            print(label[i])
            # press = cv2.waitKey(0)
            # if (press == ord('g')):
            int_classifications.append(ord(label[i]))
            flattened_image = img_roi_resized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            flattened_images = np.append(
                flattened_images, flattened_image, 0)
            print("added label!")
            # elif (press == 27):
            #     print("quit!")
            #     sys.exit()
            # else:
            #     print("skip label!")

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
