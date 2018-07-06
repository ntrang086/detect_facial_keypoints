# Facial Keypoint Detection and Real-time Filtering - PyTorch

## Introduction

Build an end-to-end facial keypoint recognition system using computer vision techniques and deep learning. Facial keypoints include points around the eyes, nose, and mouth on any face and are used in many applications, from facial tracking to emotion recognition. The system can take in any image or video containing faces and identify the location of each face and their facial keypoints. This is implemented in PyTorch.

The project will be broken up into a few main parts in four Python notebooks:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses

## Code

* Four notebooks as described above
* `data_load.py` - Code for FacialKeypointsDataset and transform steps
* `models.py` - Code for a convolutional network to predict keypoints

## Setup

### Environment

1. Clone the repository, and navigate to the downloaded folder.

```
git clone https://github.com/ntrang086/detect_facial_keypoints.git
cd detect_facial_keypoints/pytorch
```

2. Create (and activate) a new environment with Python 3.6 and the `numpy` and `pandas` packages for data loading and transformation. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n ai python=3.6 numpy pandas
	source activate ai
	```
	- __Windows__: 
	```
	conda create --name ai python=3.6 numpy pandas
	activate ai
	```
	
3. Install PyTorch and torchvision.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install -c peterjc123 pytorch-cpu
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

## Data

* Download the data from [here](https://github.com/udacity/P1_Facial_Keypoints/tree/master/data) and put them into a subdirectory named `data`. These are training and test sets of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data. 
* `detector_architectures` subdirectory: We use OpenCV's implementation of [Haar feature-based cascade classifiers]((http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)) to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades). One of these detectors has been downloaded and stored in this subdirectory.

## Run

To run any script file, use:

`python <script.py>`

To open a notebook, use:

`jupyter notebook <notebook.ipynb>`
