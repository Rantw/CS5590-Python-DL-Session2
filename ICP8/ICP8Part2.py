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


# Part 2
# load dataset
databc = pd.read_csv("Breast Cancer.csv")
#Categorize diagnosis from M or B to 0 or 1
databc['diagnosis'] = pd.Categorical(databc['diagnosis'])
databc['diagnosis'] = databc['diagnosis'].cat.codes

databc = databc.values
# Create training and testing data while removing unneeded column and pointing towards the output
X_train, X_test, Y_train, Y_test = train_test_split(databc[:,1:31], databc[:,1], test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nnbc = Sequential()
my_first_nnbc.add(Dense(30, input_dim=30, activation='relu'))
my_first_nnbc.add(Dense(1, activation='sigmoid'))

#added metrics to display accuracy
my_first_nnbc.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
my_first_nnbc_fitted = my_first_nnbc.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)


print(my_first_nnbc.summary())
print('[Loss, Accuracy]:', my_first_nnbc.evaluate(X_test, Y_test, verbose=0))
