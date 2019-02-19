# Ronnie Antwiler
# 1.Implement Naïve Bayes method using scikit-learn
# Use iris dataset available in https://umkc.box.com/s/pm3cebmhxpnczi6h87k2lwwiwdvtxyk8
# Use cross validation to create training and testing part
# Evaluate the model on testing part
# 2.Implement linear SVM method using scikit library
# Use the same dataset as above
# 3. Compare the results and report accuracy, precision, F-measure and Recall

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

# Part 1: Naive Bayes
# Get abd Separate data
irisdata = datasets.load_iris()
x = irisdata.data
y = irisdata.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Training Model
model = GaussianNB()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

# Print Accuracy, Precision, F1, and Recall scores
print('Naive-Bayes results:')
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
print('Precision:', metrics.precision_score(y_test, y_predict, average="macro"))
print('F1 score:', metrics.f1_score(y_test, y_predict, average="macro"))
print('Recall:', metrics.recall_score(y_test, y_predict, average="macro"))

# Part 2 linear SVM method
from sklearn.svm import SVC

#Get and Separate Data
irisdata2 = datasets.load_iris()
x2 = irisdata2.data
y2 = irisdata2.target
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2)

# Train Model
model2 = SVC(kernel='linear', C=1)
model2.fit(x2_train, y2_train)
y2_predict = model2.predict(x2_test)

print()
print()
print('Linear SVM results:')
print("Accuracy:", metrics.accuracy_score(y2_test, y2_predict))
print('Precision:', metrics.precision_score(y2_test, y2_predict, average="macro"))
print('F1 score:', metrics.f1_score(y2_test, y2_predict, average="macro"))
print('Recall:', metrics.recall_score(y2_test, y2_predict, average="macro"))

# Part 3
# 3. Compare the results and report accuracy, precision, F-measure and Recall
# The Linear SVM seems to score higher most times but there are a few times that the Naive Bayes. Also,
# the scores seem to be the upper 0.90s most times for both models. This would lead me to believe that
# the data set is easily separated or predicted.
