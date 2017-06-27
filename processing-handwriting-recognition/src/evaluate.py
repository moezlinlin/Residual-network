# coding:utf-8
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
from keras.utils import np_utils


def arrange_data(data, adddata, img_rows=96, img_cols=192, img_channels=3):
    """
    """
    data = data.astype('float32')
    adddata = adddata.astype('float32')
    mean = np.mean(data, axis=0)
    data -= mean
    data /= 128.
    data = data.reshape(data.shape[0], img_rows, img_cols, img_channels)
    return data,adddata

def arrange_label(label, nb_classes=60):
    """
    """
    label = np_utils.to_categorical(label, nb_classes)
    return label

def main():
    # data
    from readchars import read_part_train_image, read_test_image, get_part_train_label, get_test_label
    test_data = read_test_image(100000, 200, "resultImageRGB96")
    X_test = test_data[0]
    addX_test = test_data[1]
    Y_test = get_test_label(100000)
    X_test = arrange_data(X_test,addX_test)[0]
    addX_test = arrange_data(X_test,addX_test)[1]
    Y_test = arrange_label(Y_test)
    
    # model
    model = load_model('modeldropout/all_weights.182.h5')
    print (model.evaluate([X_test, addX_test], Y_test, verbose=0))
    result = model.predict([X_test, addX_test])
    result_label = np.argmax(result, axis=1)
    np.savetxt('allResult/result_label34.csv', result_label, delimiter=',')


if __name__ == '__main__':
    main()