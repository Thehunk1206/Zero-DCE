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
import time
from datetime import datetime
from tqdm import tqdm
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from ZeroDCE.dataset import TfdataPipeline
from ZeroDCE.zero_dce import ZeroDCE
from ZeroDCE.zero_dce_lite import ZeroDCE_lite
from ZeroDCE.losses import SpatialConsistencyLoss, \
                            ExposureControlLoss,   \
                            ColorConstancyLosss,   \
                            IlluminationSmoothnessLoss

# TODO: Add support for Pre-trained model to be loaded from checkpoint directory and continue training

tf.random.set_seed(4)
__model_type = ['zero_dce', 'zero_dce_lite']


def train(
    dataset_dir: str,
    checkpoint_dir: str,
    IMG_H:int = 400,
    IMG_W:int = 600,
    IMG_C:int = 3,
    batch_size:int = 8,
    epoch:int = 50,
    learning_rate:float = 1e-4,
    dataset_split:float = 0.05,
    logdir:str = 'logs/',
    iteration:int = 8,
    filters:int = 32,
    model_type: str = 'zero_dce',
):
    assert os.path.isdir(dataset_dir), f'Dataset directory {dataset_dir} is not a directory'
    assert os.path.exists(dataset_dir), f'Dataset directory {dataset_dir} does not exist'
    assert 0 < dataset_split < 0.3, 'Dataset split must be between 0 and 0.3'
    assert isinstance(model_type, str), 'Model Type should be a string'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # Instantiate tf.summary FileWriter
    logdir = f'{logdir}/Model/{model_type}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    train_writer = tf.summary.create_file_writer(logdir+'/train/')
    val_writer = tf.summary.create_file_writer(logdir+'/val/')

    # Instantiate dataset pipeline
    tf.print('Creating dataset pipeline...\n')
    tfdatapipeline = TfdataPipeline(
        BASE_DATASET_DIR=dataset_dir,
        IMG_H=IMG_H,
        IMG_W=IMG_W,
        IMG_C=IMG_C,
        batch_size=batch_size,
        split=dataset_split,
    )
    train_data = tfdatapipeline.data_loader(dataset_type='train')
    val_data = tfdatapipeline.data_loader(dataset_type='valid')

    # Instantiate Optimiizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Instantiating Losses
    spatial_consistency_loss = SpatialConsistencyLoss()
    exposure_control_loss = ExposureControlLoss(e=0.6, patch_size=8)
    color_constancy_loss = ColorConstancyLosss()
    illumination_smoothness_loss = IlluminationSmoothnessLoss()

    # Instantiate Zero-DCE model
    if model_type == 'zero_dce':
        tf.print('Creating Zero-DCE model...\n')
        model = ZeroDCE(
            name='DCE-Net',
            filters=filters,
            iteration=iteration,
            IMG_H=IMG_H,
            IMG_W=IMG_W,
            IMG_C=IMG_C,
        )
    elif model_type == 'zero_dce_lite':
        tf.print('Creating Zero-DCE Lite model...\n')
        model = ZeroDCE_lite(
            name='DCE-Net_lite',
            filters=filters,
            iteration=iteration,
            IMG_H=IMG_H,
            IMG_W=IMG_W,
            IMG_C=IMG_C,
        )
    else:
        raise ValueError(f'Model type {model_type} is not supported')

    # Compile the model
    tf.print('Compiling the model...\n')
    model.compile(
        optimizer=optimizer,
        spatial_consistency_loss=spatial_consistency_loss,
        exposure_control_loss=exposure_control_loss,
        color_constancy_loss=color_constancy_loss,
        illumination_smoothness_loss=illumination_smoothness_loss
    )

    tf.print(f'[INFO] Summary of model\n')
    tf.print(model.summary())

    tf.print('\n')
    tf.print('*'*60)
    tf.print('\t\t\tModel Configs')
    tf.print('*'*60)

    tf.print(
        f'\n',
        f'Model Name                : {model.name}\n',
        f'Input shape               : ({IMG_H, IMG_W, IMG_C})\n',
        f'Epochs                    : {epoch}\n',
        f'Batch Size                : {batch_size}\n',
        f'Learning Rate             : {learning_rate}\n',
        f'Dataset Split             : {dataset_split}\n',
        f'Post enhancing Iteration  : {iteration}\n',
        f'\n',
    )

    # Training the model
    tf.print('Training the model...\n')
    for e in range(epoch):
        t = time.time()

        for img in tqdm(train_data, unit='steps', desc='training...', colour='red'):
            losses = model.train_step(img)
        
        for img in tqdm(val_data, unit='steps', desc='validating...', colour='green'):
            val_losses = model.test_step(img)
        
        tf.print(f"ETA:{round((time.time() - t)/60, 2)} - epoch: {(e+1)} - loss: {losses['total_loss']}  val_loss: {val_losses['total_loss']}\n")

        tf.print('Writing logs to TensorBoard...\n')
        with train_writer.as_default():
            tf.summary.scalar('loss', losses['total_loss'], step=e+1)
            tf.summary.scalar('spatial_consistency_loss', losses['spatial_consistency_loss'], step=e+1)
            tf.summary.scalar('exposure_control_loss', losses['exposure_control_loss'], step=e+1)
            tf.summary.scalar('color_constancy_loss', losses['color_constancy_loss'], step=e+1)
            tf.summary.scalar('illumination_smoothness_loss', losses['illumination_smoothness_loss'], step=e+1)
        
        with val_writer.as_default():
            tf.summary.scalar('va_loss', val_losses['total_loss'], step=e+1)
            tf.summary.scalar('val_spatial_consistency_loss', val_losses['spatial_consistency_loss'], step=e+1)
            tf.summary.scalar('val_exposure_control_loss', val_losses['exposure_control_loss'], step=e+1)
            tf.summary.scalar('val_color_constancy_loss', val_losses['color_constancy_loss'], step=e+1)
            tf.summary.scalar('val_illumination_smoothness_loss', val_losses['illumination_smoothness_loss'], step=e+1)
        
        if (e+1)%10 == 0:
            tf.print(f'Saving model at epoch {e+1}...\n')
            model.save(f'{checkpoint_dir}/{model_type}_iter{iteration}/{model_type}_{IMG_H}x{IMG_W}_iter{iteration}_{e+1}', save_format='tf')
            tf.print(f'Saved model at epoch {e+1}\n')



def main():
    parser = argparse.ArgumentParser(description='Model training scipt for Zero-DCE models')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='Trained_Model/', help='Checkpoint directory')
    parser.add_argument('--model_type', type=str, default='zero_dce', help=f'Type of Model.should be any of: {__model_type}')
    parser.add_argument('--IMG_H', type=int, default=400, help='Image height')
    parser.add_argument('--IMG_W', type=int, default=600, help='Image width')
    parser.add_argument('--IMG_C', type=int, default=3, help='Image channels')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epoch', type=int, default=50, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dataset_split', type=float, default=0.05, help='Dataset split')
    parser.add_argument('--logdir', type=str, default='logs/', help='Log directory')
    parser.add_argument('--filters', type=int, default=32, help='Number of filters')
    parser.add_argument('--iteration', type=int, default=8, help='Post enhancing iteration')
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
        IMG_H=args.IMG_H,
        IMG_W=args.IMG_W,
        IMG_C=args.IMG_C,
        batch_size=args.batch_size,
        epoch=args.epoch,
        learning_rate=args.learning_rate,
        dataset_split=args.dataset_split,
        logdir=args.logdir,
        filters=args.filters,
        iteration=args.iteration,
    )


if __name__ == '__main__':
    main()