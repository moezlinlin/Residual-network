from keras.models import load_model
import numpy as np
from keras.utils import np_utils
# model = load_model('class_audio_classification.h5')
model = load_model('../cnn.h5')
# input image dimensions
img_rows, img_cols = 78, 100
# The CIFAR10 images are RGB.
img_channels = 3
nb_classes = 4

val_feature = np.loadtxt(open("test_train.csv"), delimiter=",", dtype=np.uint8)
print("loaded feature")
val_label = np.loadtxt(open("test_label.csv"), delimiter=",", dtype=np.uint8)
val_label = np_utils.to_categorical(val_label, nb_classes)
val_feature = val_feature.astype('float32')
mean_image = np.mean(val_feature, axis=0)
val_feature -= mean_image
val_label /= 128.
val_feature = val_feature.reshape(val_feature.shape[0], img_rows, img_cols, 3)

res = model.evaluate(val_feature, val_label)
print(res)
