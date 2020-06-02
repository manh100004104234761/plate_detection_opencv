import cv2

img = cv2.imread('test.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hue, img_saturation, img_value = cv2.split(img_hsv)

print(img_hsv)
print(img_value)

cv2.imshow('gray', img)
cv2.imshow('threshold', img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
