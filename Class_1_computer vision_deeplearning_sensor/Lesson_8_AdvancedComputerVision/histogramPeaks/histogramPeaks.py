import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load our image
# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
img = mpimg.imread('/home/jongmin/Udacity/Lesson_8_AdvancedComputerVision/histogramPeaks/warped-example.jpg')/255

def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]
    print(bottom_half.shape)
    plt.imshow(bottom_half)
    plt.show()
    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    # 각 Column마다 Row값을 더함 (axis=0) --> 세로의 픽셀값을 모두 더함
    histogram = np.sum(bottom_half, axis=0)
    print(histogram.shape)
    return histogram

# Create histogram of image binary activations
histogram = hist(img)

# Visualize the resulting histogram
plt.plot(histogram)
plt.show()