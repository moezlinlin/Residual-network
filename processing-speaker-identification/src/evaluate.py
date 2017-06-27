# coding:utf-8
import numpy as np
from keras.utils import np_utils
from keras.models import load_model


def arrange_data(data, img_rows=78, img_cols=100):
    """
    """
    data = data.astype('float32')
    mean = np.mean(data, axis=None)
    data -= mean
    data /= 128.
    data = data.reshape(data.shape[0], img_rows, img_cols,3)
    return data

def arrange_label(label, nb_classes=4):
    """
    """
    label = np_utils.to_categorical(label, nb_classes)
    return label

def main():
    # data
    X_validate = np.loadtxt('data/validation_feature.csv',
                            delimiter=',', dtype=np.uint8)
    Y_validate = np.loadtxt('data/validation_label.csv',
                            delimiter=',', dtype=np.uint8)
    X_validate = arrange_data(X_validate)
    Y_validate = arrange_label(Y_validate)
    
    # model
    model = load_model('model/class_audio_classification.h5')
    result = model.evaluate(X_validate,Y_validate)
    np.savetxt('data/validation.txt', result)


if __name__ == '__main__':
    main()