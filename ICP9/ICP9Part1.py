# Ronnie Antwiler
# 1. Using the history object in the source code, plot the loss and accuracy for both training data and
# validation data. (try to analyze if the model had the overfitting or not?)
# 2.Plot one of the images in the test data, and then do inferencing to check what will be the prediction
# 3. We had used 2 hidden layers and relu activation. Try to change the number of hidden layer and the
# activation function to tanh or sigmoid and see what happens

from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /=255.0
test_data /=255.0
#change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#creating network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Trained the model then saved it. If i wished to load model i would comment out model.fit after running the
# code one time and saving it.
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))
#model.save("IC9.h5")

# loaded model into program
#model.load_weights("IC9.h5")

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


#Part 1
# plot of accuracy values for training and testing
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy vs Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plot of loss values for training and testing
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Overfitting can be detected when there is major difference between training and testing accuracy.
# There does not appear to be a large difference between training and testing accuracy on the graphs so i would
# believe that overfitting does not occur with this amount of training.

# 2.Plot one of the images in the test data, and then do inferencing to check what will be the prediction
#model.load_weights("IC9.h5")

img = test_images[130]
test_img =img.reshape(((1,784)))
img_class = model.predict_classes(test_img)
prediction = img_class[0]
img = img.reshape((28,28))
print(img_class)
plt.imshow(img)
plt.title(prediction)
plt.show()