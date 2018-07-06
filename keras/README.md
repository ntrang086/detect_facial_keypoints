# Facial Keypoint Detection and Real-time Filtering - Keras

## Introduction

Build an end-to-end facial keypoint recognition system using computer vision techniques and deep learning. Facial keypoints include points around the eyes, nose, and mouth on any face and are used in many applications, from facial tracking to emotion recognition. The system can take in any image or video containing faces and identify the location of each face and their facial keypoints. This is implemented in Keras.

The project is divided into four main parts in one Python notebook:

__Part 1__: Investigating OpenCV, pre-processing (de-noising, blurring, edge detection, etc.), and face detection

__Part 2__: Training a Convolutional Neural Network to detect facial keypoints

__Part 3__: Putting parts 1 and 2 together to identify facial keypoints on any image or video

__Part 4__: Adding a filter using facial keypoints to an image and a video


## Code

* `CV_project.ipynb` - The main code to detect facial keypoints and do real-time filtering
* `utils.py` - Helper code to load and plot keypoints

## Setup

### Environment

1. Clone the repository, and navigate to the downloaded folder.

```
git clone https://github.com/ntrang086/detect_facial_keypoints.git
cd detect_facial_keypoints/keras
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

7. Troubleshoot if OpenCV throws an error when trying to run a laptop's camera on Linux. See [here](https://stackoverflow.com/questions/40207011/opencv-not-working-properly-with-python-on-linux-with-anaconda-getting-error-th?answertab=votes#tab-top) for more detail.

	- __Remove OpenCV__:
	```
	conda remove opencv
	```
	- __Update Anaconda__: 
	```
	conda update conda
	```
	- __Reinstall OpenCV as follows__: 
	```
	conda install --channel menpo opencv
	```

## Data

* `data` subdirectory contains the data needed for facial keypoint detection. Unzip the training and test data (in that same location). The data is also available on [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data).
* `detector_architectures` subdirectory: We use OpenCV's implementation of [Haar feature-based cascade classifiers]((http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)) to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades). Some of these detectors have been downloaded and stored in the `detector_architectures` subdirectory.

## Run

To run any script file, use:

`python <script.py>`

To open a notebook, use:

`jupyter notebook <notebook.ipynb>`


