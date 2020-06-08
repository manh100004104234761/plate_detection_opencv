import sys
import os
import numpy as np
import cv2


RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main():
    # declare empty array to store input
    int_classifications = []
    # declare empty numpy array, it will store img data
    flattened_images = np.empty(
        (0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    for label in os.listdir("train_data"):
        path_images = os.path.join("train_data", label)
        for image in os.listdir(path_images):
            path_image = os.path.join(path_images, image)
            img_roi_resized = cv2.imread(path_image, 0)
            int_classifications.append(ord(label))
            flattened_image = img_roi_resized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            flattened_images = np.append(
                flattened_images, flattened_image, 0)

    flt_classifications = np.array(int_classifications, np.float32)
    classifications = flt_classifications.reshape(
        (flt_classifications.size, 1))
    print("\ntraining complete~\n")

    np.savetxt("classifications.txt", classifications)
    np.savetxt("flattened_images.txt", flattened_images)

    return


if __name__ == "__main__":
    main()
# end if
