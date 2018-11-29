from __future__ import print_function
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Input, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
import math
from sklearn.metrics import f1_score
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
import codecs
import h5py
import matplotlib.pyplot as plt
import pickle
from keras import optimizers
import tensorflow as tf
import keras.backend as K
import sys
import json

import os
ls1=os.listdir('color')
if '.DS_Store' in ls1:
    ls1.remove('.DS_Store')
dic1={}
for idx,i in enumerate(ls1):
     dic1[i]=idx

count=0

h5f = h5py.File('variables.h5','r')
X = h5f['X'][:]
print("Reach 1")
Y = h5f['Y'][:]
print("Reach 2 \n")

batch_size = 128
num_classes = len(dic1)
epochs = 30

# input image dimensions
img_rows, img_cols = 256, 256
h = 256
w = 256
ch = 3
print("Reach 2.5 \n")
#tensor. will receive cifar10 images as input, gets passed to resize_images
img_placeholder = tf.placeholder("uint8", (None, 256, 256, 3))

#tensor. resized images. gets passed into Session()
resize_op = tf.image.resize_images(img_placeholder, (256, 256), method=0)


# create a generator for batch processing
# this gen is written as if you could run through ALL of the data
# AWS instance doesn't have enough memory to hold the entire training bottleneck in memory
# so we will call for 10000 samples when we call it
def gen(session, data, labels, batch_size):
    def _f():
        start = 0
        end = start + batch_size
        n = data.shape[0]
        max_iter = math.ceil(n/batch_size)
        while True:
            # run takes in a tensor/function and performs it.
            # almost always, that function will take a Tensor as input
            # when run is called, it takes a feed_dict param which translates
            # Tensors into actual data/integers/floats/etc
            # this is so you can write a network and only have to change the
            # data being passed in one place instead of everywhere
            for i in range(0,max_iter):
                # X_batch is resized
                X_batch = session.run(resize_op, {img_placeholder: data[start:end]})
                # X_batch is normalized
                X_batch = preprocess_input(X_batch)
                y_batch = labels[start:end]
                start += batch_size
                end += batch_size
                if start >= n:
                    # start = 0
                    # end = batch_size
                    print("Bottleneck predictions completed.")
                    # break

                yield (X_batch, y_batch)

    return _f


def create_model_resnet():
    input_tensor = Input(shape=(h, w, ch))
    model = VGG16(input_tensor=input_tensor, include_top=False, input_shape=(256, 256, 3))
    return model
print("Reach 2.8 \n")
X_train1, X_val1, y_train1, y_val1 = train_test_split(X, Y, test_size=0.3, random_state=0,shuffle=True)
print("Reach 3 \n")
print(X_train1.shape)
print(y_train1.shape)
with tf.Session() as sess:
    K.set_session(sess)
    K.set_learning_phase(1)

    model = create_model_resnet()

    train_gen = gen(sess, X_train1, y_train1, 128)
    bottleneck_features_train = model.predict_generator(train_gen(), 2000)
    print("conv to train list")
    print(bottleneck_features_train.shape)
    bftx = h5py.File('bftx_vgg16.h5', 'w')
    bftx.create_dataset('bftx',data = bottleneck_features_train)
    bftx.close()
    print("conv to train list complete")
    with open('bfty_vgg16.pkl', 'wb') as f:
        pickle.dump(y_train1, f)
    print("json train dump complete")

    val_gen = gen(sess, X_val1, y_val1, batch_size)
    bottleneck_features_validation = model.predict_generator(val_gen(), 2000)
    print("conv to val list")
    # bottleneck_features_validation_list = bottleneck_features_validation.tolist()
    print("conv to val list complete")
    bfvx = h5py.File('bfvx_vgg16.h5', 'w')
    bfvx.create_dataset('bfvx',data = bottleneck_features_validation)
    bfvx.close()
    with open('bfvy_vgg16.pkl', 'wb') as f:
        pickle.dump(y_val1, f)
    print("json val dump complete")
print("Reach 4 \n")

print("Reach End \n")