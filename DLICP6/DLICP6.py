# Ronnie Antwiler
#In class programming:
#1. Write the code to save the accuracy then plot the accuracy in TensorBoard
#2. Write the code to save the loss then plot the loss in TensorBoard
#3. Add one more hidden layer to autoencoder
#4. Adding a sparsity constraint on the encoded representations using activity_regularizer


from keras.layers import Input, Dense
from keras.models import Model
import keras
from keras import regularizers

# this is the size of our encoded representations
encoding_dim = 392
# 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# Added regularizer for part 4 of ICP
my_regularizer = regularizers.l1(10e-6)

# this is our input placeholder
input_img = Input(shape=(784,)) 

# "encoded" is the encoded representation of the input
# Added activity_regularizer for part 4 of ICP
encoded = Dense(392, activation='relu',activity_regularizer=my_regularizer)(input_img)

#additional hidden layer for Part 3 of ICP
encode2 = Dense(196, activation='relu')(encoded)
decode2 = Dense(392, activation='relu')(encode2)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(decode2)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

#seperate encoder model
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

#let's create a seperate decoder model
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Added metrics = accuracy so that Tensor Board will collect
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

#Added callback for TensorBoard
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#tensorboard --logdir C:/Users/rantw/Documents/School/CS5590/DLICP6/Graph/

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#Added Callback for Tensor Board
autoencoder.fit(x_train, x_train,
                epochs=40,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tbCallBack])

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)    


# use Matplotlib 
import matplotlib.pyplot as plt
# displaying original and reconstructed image
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()