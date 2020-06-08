import cv2
import numpy as np

GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
ADAPTIVE_THRESH_BLOCK_SIZE = 3
ADAPTIVE_THRESH_WEIGHT = 1


def preprocess(img_original):
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # img_max_contract_gray = maxmize_contrast(img_gray)

    height, width = img_gray.shape

    # img_blurred = np.zeros((height, width, 1), np.uint8)

    img_blurred = cv2.GaussianBlur(
        img_gray, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    img_thresh = cv2.adaptiveThreshold(img_blurred, 255.0,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       ADAPTIVE_THRESH_BLOCK_SIZE,
                                       ADAPTIVE_THRESH_WEIGHT)

    return img_gray, img_thresh


def maxmize_contrast(img_gray):

    height, width = img_gray.shape

    img_top_hat = np.zeros((height, width, 1), np.uint8)
    img_black_hat = np.zeros((height, width, 1), np.uint8)

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_top_hat = cv2.morphologyEx(
        img_gray, cv2.MORPH_TOPHAT, structuring_element)
    img_black_hat = cv2.morphologyEx(
        img_gray, cv2.MORPH_BLACKHAT, structuring_element)

    img_gray_plus_top_hat = cv2.add(img_gray, img_top_hat)

    img_gray_plus_top_hat = cv2.add(img_gray, img_top_hat)
    img_gray_plus_top_hat_minus_black_hat = cv2.subtract(
        img_gray_plus_top_hat, img_black_hat)

    return img_gray_plus_top_hat_minus_black_hat


def main():
    img = cv2.imread('test.jpg')
    img_gray, img_thresh = preprocess(img)

    cv2.imshow('gray', img_gray)
    cv2.imshow('threshold', img_thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
