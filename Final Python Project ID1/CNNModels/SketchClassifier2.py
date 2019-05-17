import keras
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import merge
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

img_size = 250
batch_size = 32
epochs = 1
num_classes = 250
K.set_image_dim_ordering('th')

train_path = 'C:/Users/rantw/Documents/School/CS5590/Project/training/'
val_path = 'C:/Users/rantw/Documents/School/CS5590/Project/validation/'

train_data = ImageDataGenerator(rescale=1./255)
val_data = ImageDataGenerator(rescale=1./255)

train_gen = train_data.flow_from_directory(train_path,
                                    target_size=(img_size, img_size),
                                    batch_size=batch_size,
                                    class_mode='categorical')
#color_mode='grayscale',

val_gen = val_data.flow_from_directory(val_path,
                                    target_size=(img_size, img_size),
                                    batch_size=batch_size,
                                    class_mode='categorical')
#color_mode='grayscale',

#print(train_gen.class_indices)
#print(val_gen.class_indices)

#val_ind = val_gen.class_indices
#train_ind = train_gen.class_indices

#print('Validation Folder:', set(val_ind) - set(train_ind))
#print('Training Folder:', set(train_ind) - set(val_ind))

input_shape = Input(shape=(3, img_size, img_size))

L1_conv = Conv2D(64, (15, 15), strides=(3, 3), activation='relu')(input_shape)
L1_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(L1_conv)

L2_conv = Conv2D(128, (5, 5), strides=(1, 1), activation='relu')(L1_pool)
L2_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(L2_conv)

L3_conv = Conv2D(256, (5, 5), strides=(1, 1), activation='relu')(L2_pool)
L3_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(L3_conv)

#tower A
TowerA_conv1 = Conv2D(48, (1, 1))(L3_pool)
TowerA_conv2 = Conv2D(64, (3, 3))(TowerA_conv1)

# Tower B
TowerB_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(L3_pool)
TowerB_conv = Conv2D(64, (1, 1))(TowerB_pool)

# Tower C
TowerC_conv = Conv2D(64, (3, 3))(L3_pool)

# Merge Towers
L_merged = keras.layers.concatenate([TowerA_conv2, TowerB_conv, TowerC_conv], axis=1)

L5_pool = MaxPooling2D(pool_size=(3, 3),strides=(2, 2))(L_merged)
L_flat = Flatten()(L5_pool)
L6_fullC1 = Dense(256, activation='relu')(L_flat)
L_dr = Dropout(0.5)(L6_fullC1)
L7_fullC2 = Dense(250, activation='sigmoid')(L_dr)
model = Model(input_shape, L7_fullC2)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(train_gen,
                              steps_per_epoch=train_gen.samples // batch_size,
                              validation_data=val_gen,
                              validation_steps=val_gen.samples // batch_size,
                              epochs=epochs,
                              verbose=1)
