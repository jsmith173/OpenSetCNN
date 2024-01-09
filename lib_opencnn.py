import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sklearn.metrics as sk_metrics
import tensorflow_datasets as tfds
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import datasets
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

#
def get_deterministic_model(input_shape, units, loss, optimizer, metrics):
 model = keras.Sequential([
     layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=input_shape),
     layers.MaxPooling2D(pool_size=(6, 6)),
     layers.Flatten(),
     layers.Dense(units=units, activation='softmax')
 ])

 model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
 return model
    
def get_subset(x, y, val):
 res = []
 for i in range(len(y)):
  if y[i] == val:
   res.append(x[i])
 return res  

def get_subset_all(x, y):
 for i in range(10):
  res = []
  subset = get_subset(x, y, i)
  res.append(subset)
  return res

