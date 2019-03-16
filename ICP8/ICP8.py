# Ronnie Antwiler
# 1.Use the use case in the class:
#   a. Add Dense layers to the existing code and check how the accuracy changes.
# 2. Change the data source to Breast Cancer dataset * available in the source folder and make required changes
# Breast Cancer dataset is designated to predict if a patient has Malignant (M) or Benign = B cancer

import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# Part 1
# load dataset
dataset = pd.read_csv("diabetes.csv", header=None).values
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8], test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential()

#additional layers added to see what happens
my_first_nn.add(Dense(20, input_dim=8, activation='relu'))
my_first_nn.add(Dense(10, activation='relu'))
#my_first_nn.add(Dense(5, activation='relu'))
my_first_nn.add(Dense(1, activation='sigmoid'))

#added metrics to display accuracy
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
# Display Loss and then accuracy
print('[Loss, Accuracy]:', my_first_nn.evaluate(X_test, Y_test, verbose=0))

# Original layer: Loss: 0.6598 Accuracy: 0.7031
# Another layer added : Loss: 0.5918 Accuracy: 0.6875
# second layer added: Loss: 0.5963 Accuracy: 0.7135
# The loss decreases but the accuracy does as well with another layer.
# Accuracy increases to higher level with a second additional layer however loss increases slightly.
# Perhaps there is a trade off between loss and accuracy with increasing complexity to the system. More complexity
# could lead to additional places to suffer losses