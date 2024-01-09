import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sklearn.metrics as sk_metrics
import tensorflow_datasets as tfds
import lib_opencnn as nn
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import datasets
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Model / data parameters
num_mnist_all = 10
num_classes = 2
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print('Started')

x_subset_all = nn.get_subset_all(x_train, y_train)
 
for i in range(1):
 deterministic_model = nn.get_deterministic_model(
     input_shape=input_shape,
     units=num_classes
     loss="categorical_crossentropy",
     optimizer="adam",
     metrics=["accuracy"]
 )
 
 x_train_subset = x_subset_all[i]
 y_train_subset = np.zeros((len(x_train_subset),), dtype=int)
 y_train_subset.fill(1) #yes=1, no=0
 
 deterministic_model.summary()
 deterministic_model.fit(x_train_subset, y_train_subset, batch_size=128, epochs=5, validation_split=0.1)
 deterministic_model.save('deterministic_mnist.keras')

 print('Accuracy on MNIST test set: ', str(deterministic_model.evaluate(x_test, y_test, verbose=False)[1]))

  
 
print('Finished')
