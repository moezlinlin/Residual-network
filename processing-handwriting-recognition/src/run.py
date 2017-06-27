"""
Train ResNet-34 on the run small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run.py
"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
# import h5py
# from keras.applications import imagenet_utils
import numpy as np
import resnet
import time

def process_data(pos_vec,train_set,rgb,dir):


    nb_classes = 60
    ## input image dimensions
    img_rows, img_cols = 96, 192
    ## The CIFAR10 images are RGB.
    img_channels = rgb
    # flags = [train_set, 100000 ,train_set, 100000]
    flags = [train_set, train_set]
    print("Begin loading feature")
    start = time.clock()
    from readchars import read_part_train_image, read_test_image, get_part_train_label, get_test_label
    train_data = read_part_train_image(flags[0], pos_vec, dir)
    # test_data = read_test_image(flags[1], pos_vec, dir)
    X_train = train_data[0]
    # X_test = test_data[0]
    addX_train = train_data[1]
    # addX_test = test_data[1]
    Y_train = get_part_train_label(flags[1])
    # Y_test = get_test_label(flags[3])
    end = time.clock()
    print('----------------Running time: %s Seconds----------------\n' % (end - start))
    # dataSet = np.loadtxt(open("data/test_train.csv"),
    #                      delimiter=",", dtype=np.uint8)
    # label = np.loadtxt(open("data/test_label.csv"), delimiter=",")
    # format train and test
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     dataSet, label, test_size=0.2, random_state=0)
    # Convert class vectors to binary class matrices.


    # Convert labels to categorical one-hot encoding
    print("Begin Convert labels to categorical one-hot encoding")
    start = time.clock()
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    # Y_test = np_utils.to_categorical(Y_test, nb_classes)
    end = time.clock()
    print('----------------Running time: %s Seconds----------------\n' % (end - start))

    print("Begin astype to float32")
    start = time.clock()
    X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    addX_train = addX_train.astype('float32')
    # addX_test = addX_test.astype('float32')
    end = time.clock()
    print('----------------Running time: %s Seconds----------------\n' % (end - start))

    # subtract mean and normalize
    print("Begin subtract mean and normalize")
    start = time.clock()
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    # X_test -= mean_image
    X_train /= 128.
    # X_test /= 128.
    end = time.clock()
    print('----------------Running time: %s Seconds----------------\n' % (end - start))

    print("Begin reshape")
    start = time.clock()
    print(X_train.shape[0])
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
    # X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
    print(X_train.shape)
    end = time.clock()
    print('----------------Running time: %s Seconds----------------\n' % (end - start))
    return [X_train,addX_train,Y_train]

def main(nepoch,pos_vec,train_set,rgb , dropout, interval, X_train,addX_train,Y_train):

    # define
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(
        0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    # early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger(
        'data/all_uni_resnet34_' + str(nepoch) + '_' + str(pos_vec) + '_' + str(train_set) + '_' + str(rgb) + '_' + str(dropout) + 'class.csv',
        append=True)
    checkpointer = ModelCheckpoint(filepath='allModel/'+ str(dropout) +'_weights.{epoch:02d}.h5', verbose=1)

    batch_size = 32
    nb_epoch = nepoch
    img_channels = rgb
    nb_classes = 60
    img_rows, img_cols = 96, 192
    #return
    # model:
    ## construct model
    print ("Begin construct model")
    start = time.clock()
    if nepoch == interval:
        model = resnet.ResnetBuilder.build_resnet_34(
            (img_channels, img_rows, img_cols), nb_classes, pos_vec)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    else:
        model = load_model('allModel/all_uni_class_words_classification34_' + str(nepoch-interval) + '_' + str(pos_vec) + '_' + str(train_set) + '_' + str(rgb) + '_' + str(dropout)+ '.h5')
    print (X_train.shape)
    end = time.clock()
    print('----------------Running time: %s Seconds----------------\n' % (end - start))


    ## train model
    print("Begin train model")
    start = time.clock()
    i_epoch = nepoch-interval
    model.fit([X_train, addX_train], Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0,
              shuffle=True, callbacks=[lr_reducer, csv_logger, checkpointer],
              initial_epoch = i_epoch)
    model.save('allModel/all_uni_class_words_classification34_' + str(nepoch) + '_' + str(pos_vec) + '_' + str(train_set) + '_' + str(rgb) + '_' + str(dropout) + '.h5')
    end = time.clock()
    print('----------------Running time: %s Seconds----------------\n' % (end - start))


    # # evaluate
    # print("Begin evaluate")
    # start = time.clock()
    # score = model.evaluate([X_test,addX_test], Y_test, verbose=0)
    # print("Test loss: ", score[0], "Test accuracy: ", score[1])
    # print (score)
    # end = time.clock()
    # print('----------------Running time: %s Seconds----------------\n' % (end - start))
    # print("Begin predict")
    # start = time.clock()
    # result = model.predict([X_test,addX_test])
    # #print (np.argmax(result,axis=1))
    # print (result.shape)
    # result_label = np.argmax(result, axis=1)
    # np.savetxt('result/2result_label34_'+str(nepoch)+'_'+str(pos_vec)+'_'+str(train_set)+'_'+str(rgb) + '_' + str(dropout) + '.csv', result_label, delimiter=',')
    # end = time.clock()
    # print('----------------Running time: %s Seconds----------------\n' % (end - start))

if __name__ == '__main__':
    total_start = time.clock()
    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    #
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # session = tf.Session(config=tf_config)
    # alist = [234, 200, 125, 75]
    # for key in alist:
    # pos_vec, train_set, rgb, dir
    alist = process_data(200, 100000, 3, "resultImageRGB96")
    #for i in range(40,240,40):
        # nepoch, pos_vec, train_set, rgb, interval, X_train, addX_train, Y_train, X_test, addX_test, Y_test
    main(200, 200, 100000, 3, "dropout", 200,alist[0],alist[1],alist[2])
    total_end = time.clock()
    print('----------------TotalRunning time: %s Seconds----------------\n' % (total_end - total_start))