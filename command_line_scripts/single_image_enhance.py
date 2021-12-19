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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.utils import save_img

from utils import get_model, read_image, post_enhance_iteration, plot_image_enhanced_image_amaps

def run_inference(model_path:str, image_path:str, img_h:int=200, img_w: int=300, iteration:int = 6, plot:int = 1, save_result:int = 0)->None:
    '''
    Run inference on a single resized image.
    args:
        model_path: path to tf model
        image_path: path to image file
        img_h: image height
        img_w: image width
        iteration: number of Post Ehnancing iterations
        plot: plot enhanced image. 0: no plot, 1: plot
        save_result: save enhanced image. 0: no save, 1: save
    return: None
    '''
    assert plot in [0, 1] , 'plot must be either 0 or 1'
    assert save_result in [0, 1] , 'save_results must be either 0 or 1'
    
    
    _results_dir = './output/'
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

    if plot == 1:
        a_maps_shifted_mean = (a_maps + 1)/2 # normalize alpha maps to [0,1] for visualization
        plot_image_enhanced_image_amaps(image=image, enhanced_image=enhanced_image, amaps=a_maps_shifted_mean)
    
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

