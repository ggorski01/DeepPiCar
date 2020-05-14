# Giovanna Gorski
# Lane study, what is being detected?
import cv2
import matplotlib.pyplot as plt
import numpy as np

lane = cv2.imread('carMask.jpg')
plt.imshow(lane)
plt.show()

#From BGR to RGB
lane = cv2.cvtColor(lane, cv2.COLOR_BGR2RGB)
plt.imshow(lane)
plt.show()

#From RGB to HSV

hsv_lane = cv2.cvtColor(lane, cv2.COLOR_RGB2HSV)

#My blues
lower_blue = np.array([50, 40, 0])
upper_blue = np.array([165, 255, 255])

mask = cv2.inRange(hsv_lane, lower_blue, upper_blue)
result = cv2.bitwise_and(lane, lane, mask=mask)

plt.subplot(1,2,1)
plt.imshow(mask, cmap = "gray")

plt.subplot(1,2,2)
plt.imshow(result)
plt.show()

#detect edges

edges = cv2.Canny(mask, 200, 400)

plt.subplot(121),plt.imshow(lane,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
