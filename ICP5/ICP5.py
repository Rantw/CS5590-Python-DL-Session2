# Ronnie Antwiler
# 1Apply K means clustering on the data set College.csv in the source code
# Statistics for a large number of US Colleges from the 1995 issue of US News and World Report.
# 2. Calculate the silhouette score for the above clustering

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import silhouette_score

# Find file and get data
file = "College.csv"
path = ".\\"
dataset = pd.read_csv(path + file)

# Create datasets
x = dataset.iloc[:,[4, 7, 9, 18]]
y = dataset.iloc[:-1]

# Show the different chosen data scatter graphs.
sns.FacetGrid(dataset, hue="Private", size=4).map(plt.scatter, "Grad.Rate", "Enroll")
sns.FacetGrid(dataset, hue="Private", size=4).map(plt.scatter, "Grad.Rate", "Outstate")
sns.FacetGrid(dataset, hue="Private", size=4).map(plt.scatter, "Grad.Rate", "F.Undergrad")
plt.show()

# Normalize the data pulled from college.csv
norm_x = preprocessing.normalize(x)
norm_x2 = preprocessing.normalize(x)

# Put data into clusters
nclusters = 2
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(norm_x)

# Calculate the Silhouette Score. This value is between -1 and 1 and shows consistency to a cluster. +1 means
# far away from other clusters. 0 means on the border between clusters. -1 means may be placed in wrong cluster.
# Two clusters are required to use silhouette score function.
pred = km.predict(norm_x)
score = silhouette_score(norm_x, pred, metric='euclidean')
print(score)

# Set up for loop to calculate different silhouette scores using different number of clusters.
ss = []
for i in range(2, 20):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(norm_x2)
    preds = kmeans.predict(norm_x2)
    scores = silhouette_score(norm_x2, preds, metric='euclidean')
    ss.append(scores)

# Plot silhouette scores vs number of clusters
plt.plot(range(2,20), ss)
plt.title('Silhouette Vs. Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# The best number of cluster for this data is 2 based on the silhouette values.
