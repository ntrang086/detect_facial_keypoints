[//]: # (Image References)

[image1]: ./images/obamas_with_keypoints.png "Facial Keypoint Detection"

# Facial Keypoint Detection and Real-time Filtering

## Introduction

Build an end-to-end facial keypoint recognition system using computer vision techniques and deep learning. Facial keypoints include points around the eyes, nose, and mouth on any face and are used in many applications, from facial tracking to emotion recognition. The system can take in any image or video containing faces and identify the location of each face and their facial keypoints, as shown below. The sytem can also do filtering and put sunglasses above the eyes. 

![Facial Keypoint Detection][image1]

The project is divided into a few main parts in one Python notebook:

__Part 1__ : Investigating OpenCV, pre-processing (de-noising, blurring, edge detection, etc.), and face detection

__Part 2__ : Training a Convolutional Neural Network to detect facial keypoints

__Part 3__ : Putting parts 1 and 2 together to identify facial keypoints on any image

__Part 4__ : Adding a filter using facial keypoints

__Part 5__ : Enabling the system to work on video, i.e. doing Parts 3 & 4 on video


## Setup

### Environment

1. Clone the repository, and navigate to the downloaded folder.

```
git clone https://github.com/ntrang086/detect\_facial\_keypoints.git
cd AIND-CV-FacialKeypoints
```

2. Create and activate a new environment with Python 3.5 and the `numpy` package.

	- __Linux__ or __Mac__: 
	```
	conda create --name aind-cv python=3.5 numpy
	source activate aind-cv
	```
	- __Windows__: 
	```
	conda create --name aind-cv python=3.5 numpy scipy
	activate aind-cv
	```

3. Install/Update TensorFlow.

	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system. Then install the `tensorflow-gpu` package:
	```
	pip install tensorflow-gpu==1.1.0
	```
	- Option 2: __To install TensorFlow with CPU support only__:
	```
	pip install tensorflow==1.1.0
	```

4. Install/Update Keras.

```
pip install keras -U
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.

	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```

6. Install a few required pip packages (including OpenCV).

```
pip install -r requirements.txt
```

### Data
The data is in the subdirectory `data`. Unzip the training and test data (in that same location). The data is also available on [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data).

## Run

To run any script file, use:

`python <script.py>`

To open a notebook, use:

`jupyter notebook <notebook.ipynb>`


