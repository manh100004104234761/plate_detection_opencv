import math
import cv2


class PossibleChar:

    # contour
    def __init__(self, _contour):
        self.contour = _contour

        self.rect = cv2.boundingRect(self.contour)

        [int_x, int_y, int_w, int_h] = self.rect

        self.int_rect_x = int_x
        self.int_rect_y = int_y
        self.int_rect_w = int_w
        self.int_rect_h = int_h

        self.int_rect_area = self.int_rect_w * self.int_rect_h

        self.int_center_x = (self.int_rect_x +
                             self.int_rect_x + self.int_rect_w) / 2
        self.int_center_y = (self.int_rect_y +
                             self.int_rect_y + self.int_rect_h) / 2

        self.flt_diagonal_size = math.sqrt(
            (self.int_rect_w ** 2) + (self.int_rect_h ** 2))

        self.flt_aspect_ratio = float(
            self.int_rect_w) / float(self.int_rect_h)
