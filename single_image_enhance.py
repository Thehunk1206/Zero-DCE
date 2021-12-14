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
import argparse
from time import time
from typing import Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.utils import save_img

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def read_image(path: str, img_h: int = 200, img_w:int = 300) -> Tuple[tf.Tensor, tf.Tensor]:
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
    assert isinstance(original_image, tf.Tensor) , 'original_image must be a tensor'
    assert isinstance(alpha_maps, tf.Tensor) , 'alpha_maps must be a tensor'
    assert iteration > 0 and iteration < 10, 'iteration must be between 1 and 10'

    # Check if image and alpha map has batch dimension
    if original_image.shape.rank == 4:
        original_image = tf.squeeze(original_image, axis=0)
    if alpha_maps.shape.rank == 4:
        alpha_maps = tf.squeeze(alpha_maps, axis=0)

    # get original image height and width
    h, w, _ = original_image.shape

    # Resize alpha maps to original image size
    a_maps = tf.image.resize(alpha_maps, [h,w], method=tf.image.ResizeMethod.BICUBIC)
    a_maps = (a_maps-1)/2
    for _ in range(iteration):
        original_image = original_image + (a_maps)*(tf.square(original_image) - original_image)
    
    ehnanced_original_image = tf.cast(original_image*255, dtype=tf.uint8)
    ehnanced_original_image = tf.clip_by_value(ehnanced_original_image, 0, 255)

    return ehnanced_original_image


def run_inference(model_path:str, image_path:str, img_h:int=200, img_w: int=300, iteration:int = 6, plot:int = 1, save_result:int = 0)->None:
    assert plot in [0, 1] , 'plot must be either 0 or 1'
    assert save_result in [0, 1] , 'save_results must be either 0 or 1'
    
    
    _results_dir = 'output/'
    if not os.path.exists(_results_dir):
        os.mkdir(_results_dir)

    # Get image name from path
    image_name = (image_path.split('/')[-1]).split('.')[0]

    # Get model name from model path
    model_name = (model_path.split('/')[-1])

    # load the model
    model = get_model(model_path)

    # read Image
    image, original_image = read_image(image_path, img_h, img_w)

    # run inference
    tf.print(
        "[info] running inference...."
    )
    start = time()
    enhanced_image, a_maps = model(image)
    end = time()
    tf.print(
        "[info] inference time: {} ms".format((end-start)*1000)
    )

    # Get last 3 Channels of the output a_maps i.e alpha maps for each RGB channel
    a_maps = a_maps[:,:,:,:3]
    a_maps = (a_maps + 1)/2 # normalize alpha maps to [0,1]

    
    # Sqeeze the image to get rid of the batch dimension for plotting and saving
    a_maps = tf.squeeze(a_maps, axis=0)
    image = tf.squeeze(image, axis=0)
    enhanced_image = tf.squeeze(enhanced_image, axis=0)

    if plot == 1:

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

        ax3.imshow(a_maps[:,:,0])
        # sns.heatmap(a_maps[:,:,0], robust=True,ax=ax3).axis('off')
        ax3.set_title('Alpha Map R')

        ax4.imshow(a_maps[:,:,1])
        # sns.heatmap(a_maps[:,:,1], robust=True,ax=ax4).axis('off')
        ax4.set_title('Alpha Map G')

        ax5.imshow(a_maps[:,:,2])
        # sns.heatmap(a_maps[:,:,2], robust=True,ax=ax5).axis('off')
        ax5.set_title('Alpha Map B')

        plt.show()

    # save enhanced image to disk
    if save_result == 1:
        ehnanced_original_image = post_enhance_iteration(original_image, a_maps, iteration)
        save_img(
            os.path.join(_results_dir, f'{model_name}_enhanced_{image_name}.jpg'),
            ehnanced_original_image
        )
        tf.print(
            "[info] saved enhanced image to disk"
        )

def main():
    parser = argparse.ArgumentParser(description='Single Image Enhancement')
    parser.add_argument('--model_path', type=str, required=True, help='path to tf model')
    parser.add_argument('--image_path', type=str, required=True, help='path to image file')
    parser.add_argument('--img_h', type=int, default=200, help='image height')
    parser.add_argument('--img_w', type=int, default=300, help='image width')
    parser.add_argument('--plot', type=int, default=1, help='plot enhanced image. 0: no plot, 1: plot')
    parser.add_argument('--save_result', type=int, default=0, help='save enhanced image. 0: no save, 1: save')
    parser.add_argument('--iteration', type=int, default=6, help='number of Post Ehnancing iterations')
    args = parser.parse_args()

    run_inference(args.model_path, args.image_path, args.img_h, args.img_w, args.iteration, args.plot, args.save_result)

if __name__ == "__main__":
    main()

