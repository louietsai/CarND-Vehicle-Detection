import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

from skimage.feature import hog


from sklearn.preprocessing import StandardScaler
#from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles


cars = []
notcars = []

extra_nov_images = glob.glob('train_images/non-vehicles/non-vehicles/Extras/*.png')
gti_nov_images = glob.glob('train_images/non-vehicles/non-vehicles/GTI/*.png')
gti_v_images = glob.glob('train_images/vehicles/vehicles/GTI_*/*.png')
kitti_v_images = glob.glob('train_images/vehicles/vehicles/KITTI_*/*.png')
#print(kitti_v_images)
cars.extend(gti_v_images)
cars.extend(kitti_v_images)
notcars.extend(gti_nov_images)
notcars.extend(extra_nov_images)

#for image in images:
#    print(image)
#    if 'image' in image or 'extra' in image:
#        notcars.append(image)
#    else:
#        cars.append(image)
 
#cars.append(gti_v_images)
#cars.append(kitti_v_images)

#notcars.append(extra_nov_images)
#notcars.append(gti_nov_images)
       
# Define a function to return some characteristics of the dataset 
# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    if len(car_list) == 0:
	return data_dict
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    #print(car_list)
    car_img_path = car_list[0]
    example_img = mpimg.imread(car_img_path)
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
				  #cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  cells_per_block=(cell_per_block, cell_per_block), #transform_sqrt=False, 
                                  #visualise=True, feature_vector=False)
                                  visualise=True)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

###### TODO ###########
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features




    
data_info = data_look(cars, notcars)
debug_data_look = True
if data_info != {} and debug_data_look == True:
	
	print('Your function returned a count of', data_info["n_cars"], ' cars and',   data_info["n_notcars"], ' non-cars')
	print('of size: ',data_info["image_shape"], ' and data type:',  data_info["data_type"])# Just for fun choose random car / not-car indices and plot example images   
	car_ind = np.random.randint(0, len(cars))
	notcar_ind = np.random.randint(0, len(notcars))
    
	# Read in car / not-car images
	car_image = mpimg.imread(cars[car_ind])
	notcar_image = mpimg.imread(notcars[notcar_ind])


	# Plot the examples
	fig = plt.figure()
	plt.subplot(121)
	plt.imshow(car_image)
	plt.title('Example Car Image')
	plt.subplot(122)
	plt.imshow(notcar_image)
	plt.title('Example Not-car Image')
	plt.show()

# Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
# Read in the image
image = mpimg.imread(cars[ind])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

debug_hog_image = True
if debug_hog_image == True:
	fig = plt.figure()
	plt.subplot(121)
	plt.imshow(image, cmap='gray')
	plt.title('Example Car Image')
	plt.subplot(122)
	plt.imshow(hog_image, cmap='gray')
	plt.title('HOG Visualization')
	plt.show()


car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
debug_car_features = True
if debug_car_features == True:
	if len(car_features) > 0:
    		# Create an array stack of feature vectors
    		X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    		# Fit a per-column scaler
    		X_scaler = StandardScaler().fit(X)
    		# Apply the scaler to X
    		scaled_X = X_scaler.transform(X)
    		car_ind = np.random.randint(0, len(cars))
    		# Plot an example of raw and scaled features
    		fig = plt.figure(figsize=(12,4))
    		plt.subplot(131)
    		plt.imshow(mpimg.imread(cars[car_ind]))
    		plt.title('Original Image')
    		plt.subplot(132)
    		plt.plot(X[car_ind])
    		plt.title('Raw Features')
    		plt.subplot(133)
    		plt.plot(scaled_X[car_ind])
    		plt.title('Normalized Features')
    		fig.tight_layout()
		plt.show()
	else: 
    		print('Your function only returns empty feature vectors...')
