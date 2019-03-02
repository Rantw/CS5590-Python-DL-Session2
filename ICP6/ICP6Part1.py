# Ronnie Antwiler
# 1.Identify and delete outliers for the “GarageArea” field in predicting SalePrice (House prices dataset)
# Hint: Plot GaurageArea v/s SalePrice using scatter plot
# Then identify and eliminate outliers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

# Get Data
file = "train.csv"
path = ".\\"
dataset = pd.read_csv(path + file)

x = dataset.iloc[:, [62, 80]]

# Subtract the mean value from the values for the column SalePrice. Then compare to the value of 2 standard
# deviations. Put that in a new dataset. Then repeat for GarageArea.
dfstd = x[np.abs(x.SalePrice - x.SalePrice.mean()) <= (2*x.SalePrice.std())]
dfstd2 = dfstd[np.abs(dfstd.GarageArea - dfstd.GarageArea.mean()) <= (2*dfstd.GarageArea.std())]

# Graph original and modified datasets
sns.FacetGrid(dataset, size=4).map(plt.scatter, "GarageArea", "SalePrice")
plt.title('Original Data')
sns.FacetGrid(dfstd2, size=4).map(plt.scatter, "GarageArea", "SalePrice")
plt.title('Data within 2 standard Deviations')
plt.show()
