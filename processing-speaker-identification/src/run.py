"""
Train ResNet-18 on the run small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run.py
"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import resnet


def main():
    # define
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(
        0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping('val_acc', min_delta=0.001, patience=2)
    csv_logger = CSVLogger('resnet18_class.csv')

    batch_size = 32
    nb_classes = 4
    nb_epoch = 50
    ## input image dimensions
    img_rows, img_cols = 78, 100
    ## The CIFAR10 images are RGB.
    img_channels = 3

    print("loading feature")
    dataSet = np.loadtxt(open("data/test_train.csv"),
                         delimiter=",", dtype=np.uint8)
    label = np.loadtxt(open("data/test_label.csv"), delimiter=",")
    # format train and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        dataSet, label, test_size=0.2, random_state=0)

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

    # model:
    ## construct model
    model = resnet.ResnetBuilder.build_resnet_18(
        (img_channels, img_rows, img_cols), nb_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    ## train model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True, callbacks=[lr_reducer, early_stopper, csv_logger])
    model.save('model/class_audio_classification.h5')
    print ("train finished.")

    # evaluate
    print("beigin evaluate")
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test loss: ", score[0], "Test accuracy: ", score[1])
    print (score)
    result = model.predict(X_test)
    with open('data/result.csv', 'w') as f:
        f.write(result.to)

if __name__ == '__main__':
    main()
