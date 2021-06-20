# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import cv2

# image = mpimg.imread('exit-ramp.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# kernel_size = 3
# blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# low_threshold = 1
# high_threshold = 10

# edge = cv2.Canny(blur_gray, low_threshold, high_threshold)

# plt.figure(1)
# plt.imshow(edge, cmap='Greys_r')
# plt.figure(2)
# plt.imshow(edge, cmap='gray')
# plt.show()

#doing all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
image = mpimg.imread('solidYellowLeft.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
plt.imshow(blur_gray, cmap='gray')
plt.show()


# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 90
high_threshold = 180
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

print(edges)

# Display the image
plt.imshow(edges, cmap='Greys_r')

plt.show()
