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


class ZeroDCE_lite(tf.keras.Model):

    def __init__(self, name:str = "DCE-net_lite", filters:int = 32, iteration: int = 8, IMG_H:int = 384, IMG_W:int =512, IMG_C:int = 3, **kwargs):
        super(ZeroDCE_lite, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.iteration = iteration
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_C = IMG_C

        
    
    def call(self, inputs: tf.Tensor):
        pass
    
    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        spatial_consistency_loss: tf.keras.losses.Loss,
        exposure_control_loss: tf.keras.losses.Loss,
        color_constancy_loss: tf.keras.losses.Loss,
        illumination_smoothness_loss: tf.keras.losses.Loss,
        loss_weights: dict = {
            'spatial_consistency_w': 1.0,
            'exposure_control_w': 10.0,
            'color_constancy_w': 5.0,
            'illumination_smoothness_w': 200.0
        },
        **kwargs
    ):
        super(ZeroDCE_lite, self).compile(**kwargs)
        self.optimizer = optimizer
        self.spatial_consistency_loss = spatial_consistency_loss
        self.exposure_control_loss = exposure_control_loss
        self.color_constancy_loss = color_constancy_loss
        self.illumination_smoothness_loss = illumination_smoothness_loss
        self.loss_weights = loss_weights
    
    def compute_losses(self, input_img:tf.Tensor, enhanced_img: tf.Tensor, a_maps: tf.Tensor)-> dict:
        '''
        Compute all zero reference DCE losses
        args:
            input_img: tf.Tensor, input image
            enhanced_img: tf.Tensor, enhanced image
            a_maps: tf.Tensor, Alpha maps of enhanced image
        return:
            dict, loss dictionary
        '''
        l_spa = self.loss_weights['spatial_consistency_w'] * self.spatial_consistency_loss(input_img,enhanced_img)
        l_exp = self.loss_weights['exposure_control_w'] * self.exposure_control_loss(enhanced_img)
        l_col = self.loss_weights['color_constancy_w'] * self.color_constancy_loss(enhanced_img)
        l_ill = self.loss_weights['illumination_smoothness_w'] * self.illumination_smoothness_loss(a_maps)

        total_loss = l_spa + l_exp + l_col + l_ill

        return {
            'total_loss': total_loss,
            'spatial_consistency_loss': l_spa,
            'exposure_control_loss': l_exp,
            'color_constancy_loss': l_col,
            'illumination_smoothness_loss': l_ill
        }

    @tf.function
    def train_step(self, inputs: tf.Tensor) -> dict:
        '''
        Forward pass, calculate total loss, and calculate gradients with respect to loss.
        args:
            inputs: tf.Tensor, Tensor of shape (batch_size, IMG_H, IMG_W, IMG_C)
        returns:
            loss: tf.Tensor, Tensor of shape (batch_size, 1)
        '''
        with tf.GradientTape() as tape:
            enhanced_img, a_maps = self(inputs)
            losses = self.compute_losses(inputs, enhanced_img, a_maps)
        
        # Calculate gradients
        gradients = tape.gradient(losses['total_loss'], self.trainable_variables)
        # Backpropagate gradients to update weights.
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return losses
    
    @tf.function
    def test_step(self, inputs: tf.Tensor)-> dict:
        '''
        Forward pass, calculate total loss.
        args:
            inputs: tf.Tensor, Tensor of shape (batch_size, IMG_H, IMG_W, IMG_C)
        returns:
            dict, validation loss dictionary
        '''
        enahncened_img, a_maps = self(inputs)
        val_losses = self.compute_losses(inputs, enahncened_img, a_maps)
        return val_losses

    def summary(self, plot:bool = False):
        x = tf.keras.Input(shape=(self.IMG_H, self.IMG_W, self.IMG_C))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='DCE-net_lite')
        if plot:
            tf.keras.utils.plot_model(model, to_file='DCE-net_lite.png', show_shapes=True, show_layer_names=True, rankdir='TB')
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
    model = ZeroDCE_lite(filters=32, iteration=1)
    tf.print(model.summary(plot=True))
    tf.print(model.get_config())
    y,a_maps = model(x)
    tf.print(y.shape)
    tf.print(a_maps.shape)
