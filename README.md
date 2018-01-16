## Advanced Lane Finding

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

Each step of the pipeline was saved in `output_images` and `output_transformations`. Folders like `output_images_challenge`, `challenge_images` were used to tune the pipeline.
The final video called `videos_output/improved_project_video.mp4` is the video my pipeline works well on and is the processed result of 'project_video'

For mode details refer to the [writeup](./writeup.md).


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sbatururimi/Advanced-Lane-Lines/blob/master/LICENSE)



