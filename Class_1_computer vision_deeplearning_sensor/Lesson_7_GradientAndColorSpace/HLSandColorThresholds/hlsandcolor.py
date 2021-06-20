import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('/home/jongmin/Udacity/Lesson_7_GradientAndColorSpace/HLSandColorThresholds/test6.jpg')

## Gray Scale
thresh_g = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary_g = np.zeros_like(gray)
binary_g[(gray > thresh_g[0]) & (gray <= thresh_g[1])] = 1

R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]

thresh_r = (200, 255)
binary_r = np.zeros_like(R)
binary_r[(R > thresh_r[0]) & (R <= thresh_r[1])] = 1

hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

## Saturation Sacle
thresh_s = (90, 255)
binary_s = np.zeros_like(S)
binary_s[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1

## Hue Scale
thresh_h = (15, 100)
binary_h = np.zeros_like(H)
binary_h[(H > thresh_h[0]) & (H <= thresh_h[1])] = 1



f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(binary_g, cmap='gray')
ax1.set_title('Gray', fontsize=30)
ax2.imshow(binary_r, cmap='gray')
ax2.set_title('Red', fontsize=30)
ax3.imshow(binary_s, cmap='gray')
ax3.set_title('Saturation', fontsize=30)
ax4.imshow(binary_h, cmap='gray')
ax4.set_title('Hue', fontsize=30)
plt.show()