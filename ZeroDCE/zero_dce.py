'''
MIT License

Copyright (c) 2021 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


class ZeroDCE(tf.keras.Model):

    def __init__(self, name:str = "DCE-net", filters:int = 32, iteration: int = 8, IMG_H:int = 384, IMG_W:int =512, IMG_C:int = 3, **kwargs):
        super(ZeroDCE, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.iteration = iteration
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_C = IMG_C

        self.conv1 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')
        self.concat_3_4 = tf.keras.layers.Concatenate(axis=-1)
        self.conv5 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')
        self.concat_2_5 = tf.keras.layers.Concatenate(axis=-1)
        self.conv6 = tf.keras.layers.Conv2D(self.filters, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')
        self.concat_1_6 = tf.keras.layers.Concatenate(axis=-1)
        self.a_map_conv = tf.keras.layers.Conv2D(self.iteration*3, kernel_size=(3, 3), strides=(1,1), padding='same',activation='tanh')
    
    def call(self, inputs: tf.Tensor):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x_concat_3_4 = self.concat_3_4([x3, x4])
        x_5 = self.conv5(x_concat_3_4)
        x_concat_2_5 = self.concat_2_5([x2, x_5])
        x_6 = self.conv6(x_concat_2_5)
        x_concat_1_6 = self.concat_1_6([x1, x_6])
        a_maps = self.a_map_conv(x_concat_1_6)

        # Enhancing input image iteratively 
        a_maps_splited = tf.split(a_maps, self.iteration, axis=-1)
        le_img = inputs
        for a_map in a_maps_splited:
            le_img = le_img + a_map * (tf.square(le_img) - le_img)
        return le_img

    def summary(self, plot:bool = False):
        x = tf.keras.Input(shape=(self.IMG_H, self.IMG_W, self.IMG_C))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='DCE-net')
        if plot:
            tf.keras.utils.plot_model(model, to_file='DCE-net.png', show_shapes=True, show_layer_names=True, rankdir='TB')
        return model.summary()
    
    def get_config(self):
        return {
            'filters': self.filters,
            'iteration': self.iteration
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

if __name__ == "__main__":
    x = tf.random.normal([1, 400, 600, 3])
    model = ZeroDCE(filters=32, iteration=5)
    tf.print(model.summary(plot=True))
    tf.print(model.get_config())
    y = model(x)
    tf.print(y.shape)
