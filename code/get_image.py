#!/anacoda/envs/tensorflow/python
# -- coding = 'utf-8' --
# Python Version 3.7.9 # OS Windows 10
# @time : 2021/3/19 11:25
# copyright reserved
import tensorflow as tf
from PIL import Image
import os

img_num = 100
(train_images, train_labels),(test_images, test_labels)= tf.keras.datasets.mnist.load_data(os.path.abspath(os.path.dirname(__file__)) + '/../data/mnist.npz')

for i in range(img_num):
    img = test_images[i].reshape((28, 28))
    img = Image.fromarray(img * 255)
    img = img.convert('RGB')
    img.save(r'../test_img/%d.png' % i)
