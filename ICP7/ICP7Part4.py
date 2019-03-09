# Ronnie Antwiler
# 4. Change the classifier in the given code to
# a.  KNeighborsClassifier and see how accuracy changes
# b. change the tfidf vectorizer to use bigram and see how the accuracy changes
# TfidfVectorizer(ngram_range=(1, 2 ))
# c. Put argument stop_words ='english' and see how accuracy changes

# Part 4
# Source Code
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('Score of orginal code:', score)

# Part 4a
from sklearn.neighbors import KNeighborsClassifier

twenty_train2 = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect2 = TfidfVectorizer()
X_train_tfidf2 = tfidf_Vect2.fit_transform(twenty_train2.data)
# Using KNeighbors Classifier instead of MultinomialNB
clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train_tfidf2, twenty_train2.target)

twenty_test2 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf2 = tfidf_Vect2.transform(twenty_test2.data)
predicted2 = clf2.predict(X_test_tfidf2)

score2 = metrics.accuracy_score(twenty_test2.target, predicted2)
print('Score of KNeighborsClassifer:', score2)

#Part 4b
twenty_train3 = fetch_20newsgroups(subset='train', shuffle=True)

# Change ngram range to use bigrams
tfidf_Vect3 = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf3 = tfidf_Vect3.fit_transform(twenty_train3.data)
clf3 = MultinomialNB()
clf3.fit(X_train_tfidf3, twenty_train3.target)

twenty_test3 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf3 = tfidf_Vect3.transform(twenty_test3.data)
predicted3 = clf3.predict(X_test_tfidf3)

score3 = metrics.accuracy_score(twenty_test3.target, predicted3)
print('Score using bigram:', score3)

#Part 4c
twenty_train4 = fetch_20newsgroups(subset='train', shuffle=True)

# Add a stop word English to the Vector
tfidf_Vect4 = TfidfVectorizer(stop_words='english')
X_train_tfidf4 = tfidf_Vect4.fit_transform(twenty_train4.data)
clf4 = MultinomialNB()
clf4.fit(X_train_tfidf4, twenty_train4.target)

twenty_test4 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf4 = tfidf_Vect4.transform(twenty_test4.data)
predicted4 = clf4.predict(X_test_tfidf4)

score4 = metrics.accuracy_score(twenty_test4.target, predicted4)
print('Score using Stop Word:', score4)

# Using a KNeighbors Classifier reduces accuracy over a Multinomial Classifier
# Using bigrams with a multinomial classifier decreases accuracy
# Using a stop word increases the accuracy