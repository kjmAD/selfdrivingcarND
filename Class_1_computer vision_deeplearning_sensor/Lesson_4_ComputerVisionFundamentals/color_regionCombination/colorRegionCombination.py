import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('/home/jongmin/Udacity/Lesson_4_ComputerVisionFundamentals/color_regionCombination/test.jpg')

# Grab the x and y sizes and make two copies of the image
# With one copy we'll extract only the pixels that meet our selection,
# then we'll paint those pixels red in the original image to see our selection 
# overlaid on the original.
ysize = image.shape[0]
xsize = image.shape[1]
color_select= np.copy(image)
line_image = np.copy(image)

# Define our color criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define a triangle region of interest (Note: if you run this code, 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz ;)
left_bottom = [100, 720]
right_bottom = [1200, 720]
apex = [634, 428]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
# meshgird --> meshgrid(x, y) --> x 범위내의 모든 y값을 (결국 grid 형태가 됨) 배열 형태로 만듦
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
# Y = A*XX+B 보다 크거나 작은 YY 값만 True로 하여 region_threshold에 저장 (모든 픽셀에 대해 Boolean 값 생성)
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
# Mask color selection
color_select[color_thresholds] = [0,0,0]
# Find where image is both colored right and in the region
# color_threshold =200 보다 크면서 region_threshold 내에 존재하는 YY 값에 255.0.0(Red) 적용
line_image[~color_thresholds & region_thresholds] = [255,0,0]

# Display our two output images
# 두 번째 imshow 함수가 최종 결과임
plt.imshow(color_select)
plt.imshow(line_image)

# uncomment if plot does not display
plt.show()