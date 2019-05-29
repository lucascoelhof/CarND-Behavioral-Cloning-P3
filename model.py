import argparse
import csv
import os
import time
import random

import cv2

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Lambda, Cropping2D, Conv2D, MaxPooling2D, Layer
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


import sklearn
from sklearn.model_selection import train_test_split

# PARAMETERS
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                if random.random() > 0.5:
                    images.append(center_image)
                    angles.append(center_angle)
                # flipped image
                else:
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)

            # trim image to only see section with road
            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


def main(dir_):
    if dir_ and os.path.isdir(dir_):
        os.chdir(dir_)
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    model = Sequential()
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((25, 12), (0, 0))))
    model.add(Lambda(lambda x: (x / 128.0) - 1.0))
    model.add(Conv2D(24, (12, 12), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Conv2D(36, (9, 9), activation='relu'))
    model.add(Conv2D(48, (6, 6), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(16))
    model.add(Dense(1))

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    model.compile(loss='mse', optimizer=Adam(LEARNING_RATE))

    model.summary()
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=np.math.ceil(len(train_samples) / BATCH_SIZE),
                                         validation_data=validation_generator,
                                         validation_steps=np.math.ceil(len(validation_samples) / BATCH_SIZE),
                                         epochs=EPOCHS, verbose=1)

    filename = "model_" + time.strftime("%Y%m%d-%H%M%S") + ".h5"
    model.save(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains the model.')
    parser.add_argument('dir', default=os.getcwd())

    args = parser.parse_args()
    main(args.dir)
