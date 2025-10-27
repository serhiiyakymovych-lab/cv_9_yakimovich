import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, BatchNormalization, Activation, GaussianNoise, LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from time import time
from scipy import ndimage

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

# Set the seeds for reproducibility
from numpy.random import seed
from tensorflow.random import set_seed
seed_value = 1234578790
seed(seed_value)
set_seed(seed_value)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Dataset params
num_classes = 10
size = x_train.shape[1]

print('Train set:   ', len(y_train), 'samples')
print('Test set:    ', len(y_test), 'samples')
print('Sample dims: ', x_train.shape)

cnt = 1
for r in range(3):
    for c in range(6):
        idx = np.random.randint(len(x_train))
        plt.subplot(3,6,cnt)
        plt.imshow(x_train[idx, ...], cmap='gray')
        plt.title(y_train[idx])
        cnt = cnt + 1

plt.show()

# Data normalization
x_train = x_train/255
x_test = x_test/255

x_train = (x_train - 0.5) / 0.5
x_test = (x_test - 0.5) / 0.5

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

from sklearn.utils import shuffle


datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

inputs = Input(shape=(28, 28, 1))

net = GaussianNoise(0.03)(inputs)

net = Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-5))(inputs)
net = BatchNormalization()(net)
# net = LeakyReLU(alpha=0.1)(net)
net = Activation('relu')(net)
net = MaxPooling2D((2,2))(net)
net = Dropout(0.1)(net)

net = Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(1e-5))(net)
net = BatchNormalization()(net)
# net = LeakyReLU(alpha=0.1)(net)
net = Activation('relu')(net)
net = MaxPooling2D((2,2))(net)
net = Dropout(0.1)(net)

net = Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(1e-5))(net)
net = BatchNormalization()(net)
# net = LeakyReLU(alpha=0.1)(net)
net = Activation('relu')(net)
net = Dropout(0.1)(net)

net = Flatten()(net)
net = Dense(256, activation='relu')(net)
net = BatchNormalization()(net)
net = Dropout(0.3)(net)
net = Dense(128, activation='relu')(net)
net = Dropout(0.3)(net)

outputs = Dense(10, activation='softmax')(net)

model = Model(inputs, outputs)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps=2000,
    decay_rate=0.9,
    staircase=True
)

def smooth_labels(y, factor=0.1):
    y = tf.one_hot(y, depth=10)
    y = y * (1 - factor) + (factor / 10)
    return y

y_train_smooth = smooth_labels(y_train)
y_test_smooth = smooth_labels(y_test)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

history = model.fit(
    datagen.flow(x_train, y_train_smooth, batch_size=32),
    validation_data=(x_test, y_test_smooth),
    epochs=50,
    callbacks=[reduce_lr],
    verbose=1
)

def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-',
                               epochs, h['val_accuracy'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()
        
    print('Train Acc     ', h['accuracy'][-1])
    print('Validation Acc', h['val_accuracy'][-1])
    
plot_history(history)

# def datagen(x, y, batch_size):
#     num_samples = len(y)
#     while True:
#         for idx in range(0, num_samples, batch_size):
#             x_ = x[idx:idx + batch_size, ...]
#             y_ = y[idx:idx + batch_size]
            
#             if len(y_) < batch_size:
#                 x, y = shuffle(x, y)
#                 break
            
#             # Augmentation
#             for idx_aug in range(batch_size):
#                 img = x_[idx_aug, ...]
#                 r = np.random.rand()

#                 # 🔸 Диапазон без изменений
#                 if r < 0.2:
#                     pass  # ничего не делаем

#                 # 🔸 Горизонтальный флип
#                 elif r < 0.4:
#                     img = np.fliplr(img)

#                 # 🔸 Вертикальный флип
#                 elif r < 0.5:
#                     img = np.flipud(img)

#                 # 🔸 Небольшой поворот ±10°
#                 elif r < 0.7:
#                     angle = np.random.uniform(-10, 10)
#                     img = ndimage.rotate(img, angle, reshape=False)

#                 # 🔸 Изменение яркости
#                 elif r < 0.85:
#                     img = np.clip(img * np.random.uniform(0.9, 1.1), 0, 1)

#                 # 🔸 Добавление лёгкого шума
#                 else:
#                     img = np.clip(img + np.random.normal(0, 0.02, img.shape), 0, 1)               
                    
#             yield x_, y_

# inputs = Input(shape=(28,28,1))

# net = Conv2D(16, (3,3), padding='same', kernel_regularizer=l2(1e-4))(inputs)
# net = BatchNormalization()(net)
# net = Activation('relu')(net)
# net = Dropout(0.3)(net)

# net = Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-4))(net)
# net = BatchNormalization()(net)
# net = Activation('relu')(net)
# net = Dropout(0.3)(net)

# net = Flatten()(net)
# net = Dense(128, kernel_regularizer=l2(1e-4), activation='relu')(net)
# net = Dropout(0.5)(net)

# outputs = Dense(10, activation='softmax')(net)
# model = Model(inputs, outputs)
# model.summary()

# epochs = 50
# batch_size = 64
# steps_per_epoch = len(y_train) // batch_size
# generator = datagen(x_train, y_train, batch_size)

# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# history = model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=(x_test, y_test))


# net = Flatten()(net)
# net = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(net)
# net = BatchNormalization()(net)
# net = Dropout(0.5)(net)

# outputs = Dense(10, activation='softmax')(net)