from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.callbacks import TensorBoard


batch_size = 128
num_classes = 4
epochs = 100

# input image dimensions
img_rows, img_cols = 78,  100

print('loading feature')
dataSet=numpy.loadtxt(open("code/validation_feature.csv"), delimiter=",", dtype=numpy.uint8)
print('loaded feature')
label=numpy.loadtxt(open("code/validation_label.csv"), delimiter=",")
x_train, x_test, y_train, y_test = train_test_split(dataSet, label, test_size=0, random_state=0)


# the data, shuffled and split between train and test sets


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /=255
x_test /=255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# tb = TensorBoard(log_dir='./logs', histogram_freq=0)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
pred = model.predict(x_test)
numpy.savetxt('pred.csv', pred, delimiter=',')
model.save("my_model.h5")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('cnn.h5')

