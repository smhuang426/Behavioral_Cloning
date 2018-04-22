import os
import csv
from PIL import Image
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Conv2D, MaxPooling2D, Convolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2

isFirstData = 0
turn_steer_times = 0
batch_count = 0

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if isFirstData == 0:
            isFirstData = 1
        else:
            samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def augment_train_samples(samples):
    resamples = []
    for sample in samples:
        #sample[2] = sample[2].strip()
        resamples.append(sample)
        
        if np.absolute(float(sample[3])) > 0.05:
            new_sample = sample[:]
            new_sample[3] = str(-1*float(new_sample[3]))
            new_sample[0] = '-'+new_sample[0]
            #new_sample[2] = '-'+new_sample[2].strip()
            resamples.append(new_sample)
    
    return resamples


def train_generate(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                #center
                name = batch_sample[0].split('-')
                if name[0] == '':
                    image = Image.open(name[-1])
                    image = np.asarray(image)
                    center_image = np.fliplr(image)
                else:
                    center_image = Image.open(name[-1])
                    center_image = np.asarray(center_image)

                center_image = center_image[60:140,:]
                center_image = cv2.resize(center_image, (64, 64))
                center_image = center_image/255.0-0.5
                
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def validat_generate(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0].split('\\')[-1]
                center_image = Image.open(name)
                center_image = np.asarray(center_image)
                center_angle = float(batch_sample[3])
                
                center_image = center_image[60:140,:]
                center_image = cv2.resize(center_image, (64, 64))
                center_image = center_image/255.0-0.5
                
                images.append(center_image)
                angles.append(center_angle)
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


#learning_rate = 0.001
epoch = 1
batch_size = 128

train_resamples = augment_train_samples(train_samples)

# compile and train the model using the generator function
train_generator      = train_generate(train_resamples, batch_size=batch_size)
validation_generator = validat_generate(validation_samples, batch_size=batch_size)

model = Sequential()

#model_v2
model.add(Convolution2D(8, 8,8 ,border_mode='same', subsample=(4,4), input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Convolution2D(16, 8,8 ,border_mode='same',subsample=(4,4)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 4,4,border_mode='same',subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 2,2,border_mode='same',subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Dense(1))

#model_v1
#model.add(Convolution2D(16, 8,8 ,border_mode='same', subsample=(4,4), input_shape=(64,64,3)))
#model.add(Activation('relu'))
#model.add(Convolution2D(32, 8,8 ,border_mode='same',subsample=(4,4)))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, 4,4,border_mode='same',subsample=(2,2)))
#model.add(Activation('relu'))
#model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
#model.add(Activation('relu'))
#model.add(Flatten())
#model.add(Dropout(0.5))
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(64))
#model.add(Dense(1))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer=adam)

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_resamples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=epoch,
                                     verbose=1)

# print the keys contained in the history object
print(history_object.history.keys())

model.save('model.h5')

#model = load_model('my_model.h5')
