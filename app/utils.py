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

# This file contains utility functions for the command line scripts

import os
from time import time
from typing import Tuple


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import models

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def read_image(path: str, img_h: int = 200, img_w:int = 300) -> Tuple[tf.Tensor, tf.Tensor]:
    '''
    Decode image from path and resize it to img_h x img_w.
    Returns the resized image and the original image.
    arg:
        path:str, path to image
        img_h:int, height of image
        img_w:int, width of image
    return:
        resized_image:tf.Tensor, image tensor
        original_image:tf.Tensor, original image tensor

    '''
    assert isinstance(path, str) , 'path must be a string'
    assert os.path.exists(path) , 'Image path must be a valid path'
    assert isinstance(img_h, int) , 'img_h must be an integer'
    assert isinstance(img_w, int) , 'img_w must be an integer'

    image_raw = tf.io.read_file(path)
    original_image = tf.io.decode_image(image_raw, channels=3)
    original_image = tf.cast(original_image, dtype=tf.float32)
    original_image = original_image/255.0

    resized_image = tf.image.resize(original_image, [img_h, img_w])
    resized_image = tf.expand_dims(resized_image, axis=0)

    return resized_image, original_image


def get_model(model_path: str)-> models.Model:
    '''
    Returns tf.keras model instance from model_path
    arg:
        model_path:str, path to model
    return:
        model:tf.keras.Model, model instance
    '''
    assert isinstance(model_path, str) , 'model_path must be a string'
    assert os.path.exists(model_path) , 'model_path must be a valid path'

    tf.print(
        "[info] loading model from disk...."
    )
    model = models.load_model(model_path)

    tf.print(
        "loaded model: {}".format(model)
    )
    return model


def post_enhance_iteration(original_image:tf.Tensor, alpha_maps:tf.Tensor, iteration:int = 6)-> tf.Tensor:
    '''
    Enhance the original image by iteratively applying predicted alpha maps.
    arg:
        original_image:tf.Tensor, original image tensor
        alpha_maps:tf.Tensor, alpha maps tensor
        iteration:int, number of iterations
    return:
        enhanced_image:tf.Tensor, enhanced image tensor
    '''
    assert isinstance(original_image, tf.Tensor) , 'original_image must be a tensor'
    assert isinstance(alpha_maps, tf.Tensor) , 'alpha_maps must be a tensor'
    assert iteration < 10, 'iteration must be between 1 and 10'

    if iteration == 0:
        iteration = 1
    # Check if image and alpha map has batch dimension
    if original_image.shape.rank == 4:
        original_image = tf.squeeze(original_image, axis=0)
    if alpha_maps.shape.rank == 4:
        alpha_maps = tf.squeeze(alpha_maps, axis=0)

    # get original image height and width
    h, w, _ = original_image.shape

    # Resize alpha maps to original image size
    a_maps = tf.image.resize(alpha_maps, [h,w], method=tf.image.ResizeMethod.BICUBIC)
    # a_maps = (a_maps-1)/2
    for _ in range(iteration):
        original_image = original_image + (a_maps)*(tf.square(original_image) - original_image)
    
    ehnanced_original_image = tf.cast(original_image*255, dtype=tf.uint8)
    ehnanced_original_image = tf.clip_by_value(ehnanced_original_image, 0, 255)

    return ehnanced_original_image


def plot_image_enhanced_image_amaps(image:tf.Tensor, enhanced_image:tf.Tensor, amaps:tf.Tensor):
    # Sqeeze the image to get rid of the batch dimension for plotting and saving
    if amaps.shape.rank == 4:
        amaps = tf.squeeze(amaps, axis=0)
    if image.shape.rank == 4:
        image = tf.squeeze(image, axis=0)
    if enhanced_image.shape.rank == 4:
        enhanced_image = tf.squeeze(enhanced_image, axis=0)


    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])


    ax1.imshow(image)
    ax1.set_title('Original Image')

    ax2.imshow(enhanced_image)
    ax2.set_title('Enhanced Image')

    ax3.imshow(amaps[:,:,0])
    ax3.set_title('Alpha Map R')

    ax4.imshow(amaps[:,:,1])
    ax4.set_title('Alpha Map G')

    ax5.imshow(amaps[:,:,2])
    ax5.set_title('Alpha Map B')

    plt.show()
