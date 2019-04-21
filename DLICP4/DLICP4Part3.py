# Ronnie Antwiler
#   3. predict the first 4 image of the test data. Then, print the actual label for those 4 images (label means the
#       probability associated with them) to check if the model predicted correctly or not
#   **MNIST dataset: is a data set

# Simple CNN model for CIFAR-10
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import tensorflow
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.models import load_model

K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:6000]
y_train = y_train[:6000]

labels = np.array(['airplane','automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

model = load_model('ICP4_weights.h5')

for i in range(4):
    index = i
    print(index)
    image = x_test[index]
    img = image.astype('float32')
    img /= 255
    data = np.zeros(32 * 32 * 3).reshape((1, 3, 32, 32))
    data[0]=img

    pred = model.predict(data, batch_size=1)

    num = 0.0
    iclass = 0

    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if num < pred[0][n]:
            num = pred[0][n]
            iclass = n

    if y_test[index] == iclass:
        print("Prediction [{}].".format(labels[iclass]))
    else:
        print("Prediction: [{}]".format(labels[iclass]))
        print("Incorrect: [{}]".format(labels[y_test[index][0]]))
    print()
