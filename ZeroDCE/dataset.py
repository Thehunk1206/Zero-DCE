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

from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import glob

from sklearn.model_selection import train_test_split
import tensorflow as tf


class TfdataPipeline:
    '''
    A class to create a tf.data.Dataset object from a directory of images.
    args:
        BASE_DATASET_DIR: str, the directory of the dataset
        IMG_H: int, the height of the image
        IMG_W: int, the width of the image
        IMG_C: int, the number of channels of the image
        batch_size: int, the batch size of the dataset
        split: float, the split ratio of the dataset into train, valid, and test
    '''
    def __init__(
        self,
        BASE_DATASET_DIR: str,
        IMG_H: int = 384,
        IMG_W: int = 512,
        IMG_C: int = 3,
        batch_size: int = 16,
        split: float = 0.05
    ) -> None:
        self.BASE_DATASET_DIR = BASE_DATASET_DIR
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_C = IMG_C
        self.batch_size = batch_size
        self.split = split
        self.__datasettype = ['train', 'valid', 'test']

        if not os.path.exists(BASE_DATASET_DIR):
            tf.print(f"[Error] Dataset directory {BASE_DATASET_DIR} does not exist!")
            sys.exit()
        
    def _load_and_split_dataset_files(self, path: str, split:float = 0.06)-> tuple[list,...]:
        '''
        Loads the dataset files and splits them into train, valid, and test.
        args:
            path: str, the path to the dataset directory
            split: float, the split ratio of the dataset into train, valid, and test
        returns:
            train_files: list, the list of train files
            valid_files: list, the list of valid files
            test_files: list, the list of test files
        
        '''
        assert 0.0 < split < 1.0, "Split must be between 0.0 and 1.0"

        img_files = sorted(glob.glob(f'{path}/*'))
        total_image_size = len(img_files)
        valid_image_size = int(total_image_size * split)
        test_image_size  = int(total_image_size * split)

        train_files, val_files = train_test_split(img_files, test_size=valid_image_size, random_state=12)
        train_files, test_files = train_test_split(train_files, test_size=test_image_size, random_state=12)

        return train_files, val_files, test_files

    def _read_image(self, image_path: str) -> tf.Tensor:
        '''
        Read the content of an image file.
        args:
            image_path: str, the path to the image file

        returns:
            image: tf.Tensor, the image tensor
        '''
        img_raw = tf.io.read_file(image_path)
        image = tf.image.decode_png(img_raw, channels=self.IMG_C)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[self.IMG_H, self.IMG_W], method=tf.image.ResizeMethod.BICUBIC)
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

        return image
    
    def _tf_dataset(self, image_path: list)-> tf.data.Dataset:
        '''
        Creates a tf.data.Dataset object from a list of image files.
        args:
            image_path: str, the path to the dataset directory
        return:
            dataset: tf.data.Dataset, the tf.data.Dataset object which will be consumed by the model

        '''
        dataset = tf.data.Dataset.from_tensor_slices(image_path)
        dataset = dataset.map(self._read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        dataset = dataset.shuffle(buffer_size=50).batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
    
    # A fucntion to that returns a tf.data.Dataset object given the dataset type
    def data_loader(self, dataset_type: str)-> tf.data.Dataset:
        '''
        Returns a tf.data.Dataset object given the dataset type.
        args:
            dataset_type: str, the dataset type
        returns:
            dataset: tf.data.Dataset, the tf.data.Dataset object which will be consumed by the model
        '''
        assert dataset_type in self.__datasettype, f"Dataset type {dataset_type} is not supported"

        train_images_file, val_images_file, test_images_file = self._load_and_split_dataset_files(self.BASE_DATASET_DIR, self.split)

        if dataset_type == 'train':
            dataset = self._tf_dataset(train_images_file)
        elif dataset_type == 'valid':
            dataset = self._tf_dataset(val_images_file)
        elif dataset_type == 'test':
            dataset = self._tf_dataset(test_images_file)

        return dataset




if __name__ == "__main__":
    BASE_DIR = '/home/tauhid/Desktop/Desktop/Work space/Zero DCE-net/lol_datasetv2'

    list_of_files = sorted(glob.glob(f'{BASE_DIR}/*'))

    tfdataset = TfdataPipeline(BASE_DATASET_DIR=BASE_DIR, split=0.06)

    dataset = tfdataset.data_loader(dataset_type='test')

    for image in dataset.take(2):
        tf.print(image.shape)
        tf.print(tf.reduce_min(image), tf.reduce_max(image))

        