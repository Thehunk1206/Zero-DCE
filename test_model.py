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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import models
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from ZeroDCE.dataset import TfdataPipeline


def get_model(model_path: str):
    assert isinstance(model_path, str) , 'model_path must be a string'

    tf.print(
        "[info] loading model from disk...."
    )
    model = models.load_model(model_path)

    tf.print(
        "loaded model {}".format(model)
    )
    return model


def datapipeline(dataset_path: str, imgsize: int = 352) -> tf.data.Dataset:
    assert isinstance(dataset_path, str)

    # load dataset.
    # NOTE: Always set the batch size to 1 for testing.
    tfpipeline = TfdataPipeline(
        BASE_DATASET_DIR=dataset_path, IMG_H=imgsize, IMG_W=imgsize, batch_size=1)
    test_data = tfpipeline.data_loader(dataset_type='test')

    return test_data

# A function to plot images and their corresponding enhanced images side by side in grid. 
def plot_image(img:list, enhanced_img:list):
    assert isinstance(img, list)
    assert isinstance(enhanced_img, list)
    assert len(img) == len(enhanced_img)

    # create figure with subplots
    fig = plt.figure(figsize=(12, 12))
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
    
    plt.show()


def run_test(
    model_path:str,
    dataset_path: str = 'lol_datasetv2/',
    imgsize: int = 128,
):
    assert isinstance(model_path, str), 'model_path must be a string'
    assert isinstance(dataset_path, str), 'dataset_path must be a string'
    assert isinstance(imgsize, int), 'imgsize must be an integer'
    assert os.path.exists(model_path), 'model_path does not exist'
    assert os.path.exists(dataset_path), 'dataset_path does not exist'

    # load model
    model = get_model(model_path)

    # load dataset
    test_data = datapipeline(dataset_path, imgsize)

    # run test
    results = []
    inputs = []
    for data in tqdm(test_data.shuffle(buffer_size=50).take(5)):
        enhanced_img, _ = model(data)
        enhanced_img = tf.squeeze(enhanced_img, axis=0)
        data = tf.squeeze(data, axis=0)
        results.append(enhanced_img)
        inputs.append(data)
    
    # plot results
    plot_image(inputs, results)

if __name__ == '__main__':
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
        '--imgsize',
        type=int,
        default=128,
        help='image size'
    )
    args = parser.parse_args()

    start = time()
    run_test(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        imgsize=args.imgsize
    )
    end = time()
    tf.print(
        '[info] test finished in {} seconds'.format(end - start)
    )