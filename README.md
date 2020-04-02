# Image-Recognition-Age-Detection-Using-InceptionV3

Data Files
1. img_align_celeba.zip: All the face images, cropped and aligned
2. list_eval_partition.csv: Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
3. list_bbox_celeba.csv: Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
4. list_landmarks_align_celeba.csv: Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth
5. list_attr_celeba.csv: Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative


Import libraries

import pandas as pd

import numpy as np

import cv2    

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import f1_score

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras import optimizers

from keras.models import Sequential, Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.utils import np_utils

from keras.optimizers import SGD

from IPython.core.display import display, HTML

from PIL import Image

from io import BytesIO

import base64

plt.style.use('ggplot')

%matplotlib inline
