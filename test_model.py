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

from time import time
from tqdm import tqdm
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import models

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from ZeroDCE.dataset import TfdataPipeline

from command_line_scripts.utils import get_model

tf.random.set_seed(42)

# def get_model(model_path: str):
#     assert isinstance(model_path, str) , 'model_path must be a string'

#     tf.print(
#         "[info] loading model from disk...."
#     )
#     model = models.load_model(model_path, compile=False)

#     tf.print(
#         "loaded model {}".format(model)
#     )
#     return model


def datapipeline(dataset_path: str, img_h: int = 128, img_w:int = 256) -> tf.data.Dataset:
    assert isinstance(dataset_path, str)

    # load dataset.
    # NOTE: Always set the batch size to 1 for testing.
    tfpipeline = TfdataPipeline(
        BASE_DATASET_DIR=dataset_path, IMG_H=img_h, IMG_W=img_w, batch_size=1)
    test_data = tfpipeline.data_loader(dataset_type='test')

    return test_data

# A function to plot images and their corresponding enhanced images side by side in grid. 
def plot_image(img:list, enhanced_img:list, model_name:str, save_fig:bool=True):
    assert isinstance(img, list)
    assert isinstance(enhanced_img, list)
    assert len(img) == len(enhanced_img)

    # create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=2, ncols=len(img), figure=fig)

    for i in range(len(img)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img[i])
        ax.title.set_text('Original Image')
        ax.axis('off')
        ax = fig.add_subplot(gs[i+len(img)])
        ax.imshow(enhanced_img[i])
        ax.title.set_text('Enhanced Image')
        ax.axis('off')
    plt.tight_layout()
    plt.suptitle(f'Original vs Enhanced Image: {model_name}')
    if save_fig:
        plt.savefig(f'image_assets/test_image_plot_{model_name}.png')
    plt.show()


def run_test(
    model_path:str,
    dataset_path: str = 'lol_datasetv2/',
    img_h: int = 128,
    img_w: int = 256,
    save_plot: int = 0,
    load_random_data: int = 0
):
    assert isinstance(model_path, str), 'model_path must be a string'
    assert isinstance(dataset_path, str), 'dataset_path must be a string'
    assert isinstance(img_h, int), 'img_h must be an integer'
    assert isinstance(img_w, int), 'img_w must be an integer'
    assert save_plot in [0, 1], 'save_plot must be either 0 or 1'
    assert load_random_data in [0, 1], 'load_random_data must be either 0 or 1'
    assert os.path.exists(model_path), 'model_path does not exist'
    assert os.path.exists(dataset_path), 'dataset_path does not exist'

    model_name = model_path.split('/')[-2]

    # load model
    model = get_model(model_path)

    # load dataset
    if load_random_data == 1:
        test_data = tf.unstack(tf.random.normal([100, 1, img_h, img_w, 3]), axis=0)
    else:
        test_data = datapipeline(dataset_path, img_h=img_h, img_w=img_w)
    

    # run test
    results = []
    inputs = []
    times = []
    for data in tqdm(test_data):
        start = time()
        enhanced_img, _ = model(data)
        end = time() - start
        enhanced_img = tf.squeeze(enhanced_img, axis=0)
        data = tf.squeeze(data, axis=0)
        results.append(enhanced_img)
        inputs.append(data)
        times.append(end)

    average_time = sum(times[2:])/len(times[2:])
    # plot results
    tf.print(f'Average inference time: {round(average_time, 2)*1000} ms')
    if save_plot == 1:
        plot_image(img=inputs[:5], enhanced_img=results[:5], model_name=model_name, save_fig=save_plot)

def main():
    parser = argparse.ArgumentParser(
        description='Test model on test dataset'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='path to the saved model folder'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='lol_datasetv2/',
        help='path to the dataset'
    )
    parser.add_argument(
        '--img_h',
        type=int,
        default=128,
        help='image height'
    )

    parser.add_argument(
        '--img_w',
        type=int,
        default=256,
        help='Image width'
    )

    parser.add_argument(
        '--save_plot',
        type=int,
        default=0,
        help='save plot of original vs enhanced image. 0: no, 1: yes'
    )

    parser.add_argument(
        '--load_random_data',
        type=int,
        default=0,
        help='load random data. 0: no, 1: yes'
    )
    args = parser.parse_args()

    run_test(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        img_h=args.img_h,
        img_w=args.img_w,
        save_plot=args.save_plot,
        load_random_data=args.load_random_data
    )

if __name__ == '__main__':
    main()