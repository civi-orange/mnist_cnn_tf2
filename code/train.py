#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/3/18 17:15
# copyright reserved

import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import config


class CNN(object):
    def __init__(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.image_shap))
        model.add(layers.MaxPool2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model

class DataSource(object):
    def __init__(self):
        data_path = os.path.abspath(os.path.dirname(__file__))+'/../data/mnist.npz'
        (train_images, train_labels),(test_iamges, test_labels) = datasets.mnist.load_data(data_path)
        train_images = train_images.reshape((60000,28,28,1))
        test_iamges = test_iamges.reshape((10000,28,28,1))
        train_images, test_iamges = train_images/255.0,test_iamges/255.0

        self.train_iamges,self.train_labels = train_images, train_labels
        self.test_images,self.test_labels = test_iamges,test_labels

class Train(object):
    def __init__(self):
        self.cnn =CNN()
        self.data = DataSource()

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path,
                                                           save_weights_only=True,
                                                           verbose=1,
                                                           save_freq=5)
        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_iamges,
                           self.data.train_labels,
                           epochs=5,
                           callbacks=[save_model_cb])
        test_loss,test_acc = self.cnn.model.evaluate(self.data.test_images,
                                                     self.data.test_labels)
        print("acc: %.4f, picture number: %d" % (test_acc,len(self.data.test_labels)))

if __name__ =="__main__":
    app = Train()
    app.train()