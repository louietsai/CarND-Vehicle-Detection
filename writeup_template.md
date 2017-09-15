

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
*  normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_1.png
[image1_1]: ./output_images/notcar_1.png
[image2]: ./output_images/car_1_hig.png
[image3]: ./output_images/sliding_w_step1.png
[image4]: ./output_images/sliding_w_step2.png
[image4_1]: ./output_images/sliding_w_step3.png
[image4_2]: ./output_images/window_boxes.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4


**###Histogram of Oriented Gradients (HOG)**

The code for this step is contained in lines #474 through #498 of the file called `VehicheDetect.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

![alt text][image1_1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

**####2.  HOG parameters.**

I tried various combinations of parameters. found that small orientation may not be good for accuracy, and too small pixels_per_cell makes too larger data size.
In the end, I used below parameters.
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block

**####3.  trained a classifier using  selected HOG features ).**
The code for this step is contained in lines #500 through #519 of the file called `VehicheDetect.py`.  
I trained a linear SVM using normalized combined features including spatial, hist, and hog features. I used 80% of data as training data set, and 20% of data as testing data set.
the accuracy rate is  0.9868.

**###Sliding Window Search**
The code for this step is contained in lines #366 through #417 of the file called `VehicheDetect.py`. 
There are several different stages for my sliding window search.

1. no tracking object, and search all portion of the road like below diagram.

![alt text][image3]

2. with tracking object, keep tracking the object  and search for the three corners to find the cars which just appear 

![alt text][image4]

![alt text][image4_1]


Ultimately I searched on two scales using LUV HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4_2]
---

**### Video Implementation**

Here's a [link to my video result](./project_video_output.avi)


I recorded the positions of positive detections in each frame of the video.  
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.
The code for this step is contained in lines #419 through #431 of the file called `VehicheDetect.py`.
I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected.  


---

**###Discussion**
For tracked car in track_list, I didn't remove the tracked car from the list even if I can't find any car around the previous tracked car. Therefore, there will be still a bounding box if the car just disappear in the next frame.
it may not happen in the video, but it happen if we use some none consecutive test images.
will try to improve later

 

