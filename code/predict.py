#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/3/19 9:57
# copyright reserved
import tensorflow as tf
from PIL import Image
import numpy as np
from train import CNN


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.0
        x = np.array([img])
        y = self.cnn.model.predict(x)

        print(image_path)
        print(y[0])
        print('->predict digi', np.argmax(y[0]))
        print('----------------------')


if __name__ == '__main__':
    app = Predict()
    app.predict('../test_img/89.png')
    app.predict('../test_img/94.png')
    app.predict('../test_img/97.png')
