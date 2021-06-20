## My Project Writeup
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./output_images/02_undistort_images/signs_vehicles_xygrad.jpg "Road Transformed"
[image3]: ./output_images/03_binary_images/signs_vehicles_xygrad.jpg "Binary Example"
[image4]: ./output_images/04_perspective_transform/signs_vehicles_xygrad.jpg "Warp Example"
[image5]: ./output_images/05_detect_and_fitting/signs_vehicles_xygrad.jpg "Fit Visual"
[image6]: ./output_images/06_07_complete_images/signs_vehicles_xygrad.jpg "Output"
[image7]: ./camera_cal/calibration1.jpg "Raw chessboard image"
[image8]: ./output_images/01_camera_cal/calibration1.jpg "Undistorted chessboard"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### 01. Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "/CarND-Advanced-Lane-Lines/my_project_solutionipynb".  

First, I find corner points in each images using 'cv2.findChessboardCorners()'. Found points and generated object points are appended in each images. And this parameters are used as parameter of 'cv2.calibrateCamera()'. cv2.calibrateCamera() function returns 'mtx' and 'dist' values that are camrea calibration parameter.These values are saved as pickle files for future reuse.

Then, I undistort each images using 'mtx' and 'dist'. I additionally transformed the undistorted images into perspective. The output images can be viewed in the '/output_images/01_camera_cal' folder.

![alt text][image7] 

![alt text][image8]

### 02. Distortion correction to raw images

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
* In this pipeline, raw images(/test_images) will be undistorted
* Input images PATH = /test_images
* Output images PATH = /output_images/02_undistort_images

![alt text][image2]

### 03. Create binary image

I used a combination of color and gradient thresholds to generate a binary image (L channel from HLS, V channel from HSV and gradient from sobel-x). The code for my creating binary image includes a function called `binary_threshold()`.
* *Using color & gradient method
* Input images PATH = /test_images
* Output Images PATH = /output_images/03_binary_images

![alt text][image3]

### 04. Perspective Tansform (bird-eye view)

The code for my perspective transform includes a function called `warpImage()`.   The `warpeImage()` function takes as inputs an image (`img`) source (`src`), destination (`dst`) and warp method (`unwarp`) which decide warp or reversely warp image. I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32(
       [[(img_size[0]/2)-65, (img_size[1]/2)+100],
        [(img_size[0]/2)+65, (img_size[1]/2)+100],
        [(img_size[0]*5/6)+60, img_size[1]],
        [(img_size[0]/6)-60, img_size[1]]])
    
    dst = np.float32(
       [[(img_size[0]/4), 0],
        [(img_size[0]*3/4), 0],
        [(img_size[0]*3/4), img_size[1]],
        [(img_size[0]/4), img_size[1]]])
```

This resulted in the following source and destination points:
* perspective transform image as ROI
* Before perpective transforming, image is undistorted and binarized
* Input images PATH = /test_images
* Output Images PATH = /output_images/04_perspective_transform

![alt text][image4]

### 05. Detecting lane and Fitting curve

Using the output image through the functions described above, a pixel that may correspond to a curve was detected, and a 2-order polynomial fitting was applied to determine the final lane line. The code of function called `fit_polynomial()`.
* Detect left and light lane line pixcels using histogram
* Fitting a polynomial to lane lines
* Input images PATH = /test_images
* Output Images PATH = /output_images/05_pixelandfitting

![alt text][image5]

### 06 & 07 & 08. Curvature, Offset and Unwarp

The function for calculate curvature and center offset of the lanes called `complete_img()`. I take center of both lanes(left/right) for calculate position of vehicle. My `complete_img()` function have following sequence.
* Init image -> Undistort -> Binary -> Warp -> Find lanes -> Unwarp
* Calculate curvature and offset of the lanes
* Unwarp processed image to original perspective image.
* Input images PATH = /test_images
* Output Images PATH = /output_images/06_07_complete_images

![alt text][image6]

---

### 09. Apply to video

My code explained above is also used to video processing.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

Basically, the algorithm was implemented using the techniques learned in the advanced computer vision lecture. However, it is unstable in areas where the color of the road changes and where one lane is blurred.
In order to solve this problem in the future, it is necessary to find the variable 'src' and 'dist' points in the perspective transform. Using the hough technique learned in the previous lecture will lead to more accurate perspective transformation in various situations. And when creating a binary image, it is necessary to consider the threshold value and color separation.
