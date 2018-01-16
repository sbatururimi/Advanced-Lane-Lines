## Writeup 

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

[image0]: ./output_images/chessboard_corners.jpg "Chessboard"
[image1]: ./output_images/undistortion_chessboard.jpg "Undistorted"
[image2]: ./output_images/undistortion.jpg "Road Transformed"
[image3]: ./output_images/color_gradient_thresholds.jpg "Binary Example"
[image4]: ./output_transformations/yellow.jpg "Yellow"
[image5]: ./output_transformations/range_white.jpg "White range"
[image6]: ./output_transformations/range_yellow.jpg "Yellow range"
[image7]: ./output_images/color_gradient_thresholds.jpg "Gradient threshold"
[image8]: ./output_images/perspective_transform.jpg 'Perspective transform'
[image9]: ./output_images_challenge/failed_sliding_window_project_image_1.jpg "Failed line finding"
[image10]: ./output_images/map_line.jpg "Road lane lines"

<!-- [image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video" -->

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

Using a set of chessboard images, I have computed the camera calibration matrix and distortion coefficients.
By counting the number of corners in any given row I got a value  nx. Similarly, I have counted the number of corners in a given column and stored that in ny. It was important to keep in mind that "corners" are only points where two black and two white squares intersect, in other words, to only count inside corners, not outside corners.
We obtain a 9x6 chessboard.


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Using the distortion coefficients `dist` and the calibration matrix `mtx` obtained during the camera calibration step, I have applied the `cv2.undistort()` function to the road image. The undistortion is mostly visible by looking at the top right bush. The camera calibration step is done only once and in order to save this computation I have saved `dist` and `mtx` to the disk, so that when we need, we can load them again without going through the calibbration step.



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Here's an example of my output for this step:

![alt text][image3]

To achieve that I have a combination of color extraction and gradiend thresholds. 
1) For the color extraction, I have converted the image to the HLS space and then applied a threshold on each channel to retrieve white and yellow lines.
```
def hls_select_color_lines(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[((l_channel >= 200) & (l_channel <= 255)) | 
                  ((h_channel >= 10) & (h_channel <= 40) & (s_channel >= 100) & (s_channel <= 255))] = 1
    return binary_output
 ```
 The lower and upper bound were determined using a color picker tool.
 ![alt text][image4]
 ![alt text][image5]
 ![alt text][image6]

 *Note*  This step was taken from my previous Lane-Lines detection project.

 2) For the gradient threshold, I have used the Sobel operator for the x orientation in the HLS color space, particulary by applying the gradient operator to the L channel. Here's an example of my output for this step:

 ![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`.  The `corners_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) points.  I chose the hardcode the source and destination points in the following manner:

```python
corners = [(190, 718), (580, 460), (704, 460), (1115, 718)]
vertices = np.array([corners], dtype=np.int32)
...
src = np.float32(vertices)
dst = np.float32([[290, 718], [290, 0], [990, 0], [990, 718]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 718      | 290, 718        | 
| 580, 460      | 290, 0      |
| 704, 460      | 990, 0      |
| 115, 718      | 990, 718        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After a lot of tuning and experiments, I finally come with a `LaneDetector` class performing the road lanes detection. This class contains 2 instances of a `Line` class that helps to identify the left and right lane lines. 
After the preprocessing steps (color extraction, color/gradient thresholds, undistortions, perspective transformation), I then do the following:
1) we already have 2 detected lanes from the previous step, then I use the previously found windows to search the lines in the next frmame within a margin around the previous line position
2) if we didn't have any previously detected lines, we apply the windows search method by using the peaks detection in the histogram. Here, I have made small tune, so that we check for the left peak in the first 1/3 of the width of the image, and for the right lane, we look for a peak in the third part of the width of the image.
This helps as to fix problems like the following:

![alt text][image9]

3) In the next step, I compute the radius of curvature for the left and right lanes and then perform a simple sanity test by comparing if the found radius of curvature for the left and right lanes are similare enough. I consider them to be similar if the difference is no more than 1500 meters.
4) If the sanity check passes, then we consider the lines to be detected and draw them on the image.

![alt text][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For the curvature radius, I used a pixels to meter conversion scaling parameters
```python
self.ym_per_pix = 30/720 # meters per pixel in y dimension
self.xm_per_pix = 3.7/1280 # meters per pixel in x dimension
```
 We can assume that if we're projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide. Or, if we prefer to derive a conversion from pixel space to world space in our own images, we should compare our images with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each.

 ```python
     def radius_curvature(self, leftx, lefty, rightx, righty, ploty):
        # define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image, i.e closest to our vehicle
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/1280 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)

        # new radius of curvature
        left_curverad_world = ((1 + (2*left_fit_cr[0] * y_eval * self.ym_per_pix + 
                                     left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad_world = ((1 + (2*right_fit_cr[0] * y_eval * self.ym_per_pix 
                                      + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad_world, right_curverad_world
 ```


 This code can be found inside the `LaneDetector` class. It is important to note that a big value signify that the lanes are almost vertical, so that the radius strives for infinity.

 For the position of the vehicle, I have taken the bottom x coordinates of the left and right curves, reprsenting the road lane lines, then their center and after that, I have compared it to the center of the image as follow:

 ```python
midpoint_lanes = (self.left_line.current_fit[-1] + self.right_line.current_fit[-1]) / 2
...
offset = midpoint_lanes - midpoint_image
 ```
 I have then converted it to meters using the `xm_per_pix` scaling factor used to compute the radius of curvature. A negative value of the offset signifying that the car is positioned left compared to the center and a positive value - right to the center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in my code in  `LaneDetector` class in the method `map_lane()`.  Here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./videos_output/improved_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have tested my pipeline on the challenge video and directly discovered some problems when:
- the road has different colors
- when we have also a lot of shadows on the road but with different degree of lightning.

We could apply a bunch of different improvements:
1) compute if the distance between lines is correct, if not we use the previously successfull lanes. We could do it by considering several points on each lane at some interval, and computing the distance between them. A difference within a certain margin would signify that the distance is correct.
2) we should average the lane detection, for example by using a bunch of 5 lanes and when reaching this bound, starting by adding the newly lanes to the head(using a First-In First-out structure)
3) checking the reasonability of the radius of curvature. A big value, signifying that lanes are almost vertical. We could check that by using the bird-eye view image and then averaging the angle of each lane.
4) checking whether lanes are parallel by using a similar appraoch to 1), i.e by taking 2 poinst on each lane, forming a line and comparing them to line, formed by the 2 points of the other lanes. If the angle between them is within a certain angle, we should consider the lane lines as parallel and we then repeat this step for several points after some intervale on each lane(no necessity to check each 2 points, we can average that).
5) when the histogram method fails for the windows method, we could use convolutions. Also applying a adaptive gradient and color threshold could be interesting. That is tuning the threshold at each step by finding the best one from n-previous success frames.

### Experiments
I have also played with the LUV and LAB color channel on a image extracted from the challenge video as well as added a simple distance check between lines in the sanity check. These can be found under the section *Challenge video*
