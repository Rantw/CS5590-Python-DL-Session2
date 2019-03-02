# Ronnie Antwiler
# 2. Create Multiple Regression for the dataset (Weather dataset)
# Evaluate the model using RMSE and R2 score.
# Weather dataset: https://umkc.box.com/s/60yr8e5p0x772ggtvfmaqysswjlph5y8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

train = pd.read_csv('weatherHistory.csv')

# remove null values


# Create datasets x and y
y = train.iloc[:,[2]]
x = train.drop(['Temperature (C)', 'Loud Cover', 'Daily Summary'], axis=1)

# Convert Strings to numerical values
x['Precip Type'] = pd.get_dummies(x['Precip Type'], drop_first=True)
x['Summary'] = pd.get_dummies(x['Summary'], drop_first=True)

# split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)

# Train model
lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)

pred = model.predict(x_test)

#Check how well model preforms
print("R^2 is: ", model.score(x_test, y_test))
print('RMSE is: ', mean_squared_error(y_test, pred))

#Print predicted vs actual
plt.scatter(pred, y_test, alpha=.75, color='b')
plt.xlabel('Predicted Temperature')
plt.ylabel('Actual Temperature')
plt.title('Linear Regression Model')
plt.show()