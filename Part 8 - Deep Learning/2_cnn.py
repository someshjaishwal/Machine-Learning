#!/usr/bin/env python3

# convolution neural network

# necessray libraries
from keras.models import Sequential       # sequential way of creating NN
from keras.layers import Convolution2D   # convolution layer having feature maps 
from keras.layers import MaxPooling2D    # pooling layer having pooled feature maps
from keras.layers import Flatten         # flattening cells of pooled fea map to a vec
from keras.layers import Dense           # adding fully connected layers to NN

# Part - 1 : Initialising the CNN
classifier = Sequential()

# Step-1 Convolution Layer
classifier.add(Convolution2D(32,3,3, input_shape = (64, 64, 3), activation = 'relu')) # deprecated

# Step-2 Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step-3 Flattening Layer
classifier.add(Flatten())

# Step-4 Full connection ( hidden layers)
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part - 2 : Processing images  and fitting CNN to training set

from keras.preprocessing.image import ImageDataGenerator

# make augmentors to generate augmented dataset
train_augmentor = ImageDataGenerator(
                     rescale = 1./255,  # rescale RGB coeff b/w 0 & 1 by multiplying with a factor 1/255
                     shear_range = 0.2,
                     zoom_range = 0.2,
                     horizontal_flip = True)

test_augmentor  = ImageDataGenerator(rescale = 1./255)

# generate augmented dataset
training_set = train_augmentor.flow_from_directory(
                     directory = 'dataset/training_set',
                     target_size = (64,64),
                     batch_size = 32,
                     class_mode = 'binary')
                     #save_to_dir = 'dataset/augmented_training_set')

test_set = test_augmentor.flow_from_directory(
                     directory = 'dataset/test_set',
                     target_size = (64,64),
                     batch_size = 32,
                     class_mode = 'binary')
                     #save_to_dir = 'dataset/augmented_test_set')

# fit CNN to generated data
classifier.fit_generator( # Trains the model on data generated batch-by-batch
                    training_set,
                    samples_per_epoch = 8000, # see steps_per_epoch
                    epochs = 25,
                    validation_data = test_set,
                    nb_val_samples = 2000,
                    use_multiprocessing = True) # validation_steps = 2000
