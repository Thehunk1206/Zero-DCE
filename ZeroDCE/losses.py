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

# Highly Referenced from the orignal Pytorch Implementation of the ZeroDCE: https://github.com/Li-Chongyi/Zero-DCE_extension

class SpatialConsistencyLoss(tf.keras.losses.Loss):
    '''
    The spatial consistency loss encourages spatial coherence of the enhanced image through
    preserving the difference of neighboring regions between the input
    image and its enhanced version.

    '''
    def __init__(self, **kwargs):
        super(SpatialConsistencyLoss, self).__init__(**kwargs)
        self.kernel_left    = tf.constant([[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.kernel_right   = tf.constant([[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32)
        self.kernel_up      = tf.constant([[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.kernel_down    = tf.constant([[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32)

    def call(self, inp_image: tf.Tensor, enh_image: tf.Tensor)->tf.Tensor:
        '''
        Args:
            inp_image: The input image tensor.
            enh_image: The Low Light Enhanced image tensor.
        Returns:
            The spatial consistency loss.
        '''
        inp_image_mean = tf.reduce_mean(inp_image, axis=-1, keepdims=True)
        enh_image_mean = tf.reduce_mean(enh_image, axis=-1, keepdims=True)

        inp_image_pooled = tf.nn.avg_pool2d(inp_image_mean, ksize=4, strides=4, padding='VALID')
        enh_image_pooled = tf.nn.avg_pool2d(enh_image_mean, ksize=4, strides=4, padding='VALID')

        D_inp_left  = tf.nn.conv2d(inp_image_pooled, self.kernel_left, strides=[1, 1, 1, 1], padding='SAME')
        D_inp_right = tf.nn.conv2d(inp_image_pooled, self.kernel_right, strides=[1, 1, 1, 1], padding='SAME')
        D_inp_up    = tf.nn.conv2d(inp_image_pooled, self.kernel_up, strides=[1, 1, 1, 1], padding='SAME')
        D_inp_down  = tf.nn.conv2d(inp_image_pooled, self.kernel_down, strides=[1, 1, 1, 1], padding='SAME')

        D_enh_left  = tf.nn.conv2d(enh_image_pooled, self.kernel_left, strides=[1, 1, 1, 1], padding='SAME')
        D_enh_right = tf.nn.conv2d(enh_image_pooled, self.kernel_right, strides=[1, 1, 1, 1], padding='SAME')
        D_enh_up    = tf.nn.conv2d(enh_image_pooled, self.kernel_up, strides=[1, 1, 1, 1], padding='SAME')
        D_enh_down  = tf.nn.conv2d(enh_image_pooled, self.kernel_down, strides=[1, 1, 1, 1], padding='SAME')

        D_left  = tf.square(D_inp_left - D_enh_left)
        D_right = tf.square(D_inp_right - D_enh_right)
        D_up    = tf.square(D_inp_up - D_enh_up)
        D_down  = tf.square(D_inp_down - D_enh_down)

        return tf.reduce_mean(D_left + D_right + D_up + D_down)
    
    def get_config(self):
        return super(SpatialConsistencyLoss, self).get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ExposureControlLoss(tf.keras.losses.Loss):
    '''
    To restrain under-/over-exposed regions,
    The authors of the paper designed an exposure control loss to control the exposure
    level. The exposure control loss measures the distance between the
    average intensity value of a local region to the well-exposedness
    level E. The value of E is emperically set to 0.6 in the paper
    '''
    def __init__(self, e:float = 0.6, patch_size: int = 16, **kwargs):
        super(ExposureControlLoss, self).__init__(**kwargs)
        assert e > 0.0 and e <= 1.0, 'The Value of e must be between 0.0 and 1.0'

        self.e = e
        self.patch_size = patch_size

    def __call__(self, enhanced_img: tf.Tensor)-> tf.Tensor:
        '''
        Args:
            enhanced_img: The Low Light Enhanced image tensor.
        Returns:
            The exposure control loss.
        '''
        enhanced_img        = tf.reduce_mean(enhanced_img, axis=-1, keepdims=True)
        enhanced_img_pooled = tf.nn.avg_pool2d(enhanced_img, ksize=self.patch_size, strides=self.patch_size, padding='VALID')

        D = tf.reduce_mean(tf.square(enhanced_img_pooled - self.e))
        return D
    
    def get_config(self):
        config = super(ExposureControlLoss, self).get_config()
        config.update({
            'e': self.e,
            'patch_size': self.patch_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ColorConstancyLosss(tf.keras.losses.Loss):
    '''
    By Following the Gray-World color constancy hypothesis that color in each sensor 
    channel averages to gray over the entire image, the authors of this paper designed 
    a color constancy loss to correct the potential color deviations in the enhanced 
    image and also build the relations among the three adjusted channels.
    '''
    def __init__(self, **kwargs):
        super(ColorConstancyLosss, self).__init__(**kwargs)
    
    def __call__(self, enhanced_img: tf.Tensor)-> tf.Tensor:
        '''
        Args:
            enhanced_img: The Low Light Enhanced image tensor.
        Returns:
            The color constancy loss.
        '''
        mean_rgb = tf.reduce_mean(enhanced_img, axis=(1,2), keepdims=True)
        mr, mg, mb = tf.split(mean_rgb, 3, axis=-1)
        Drg = tf.square(mr - mg)
        Drb = tf.square(mr - mb)
        Dbg = tf.square(mb - mg)

        return tf.squeeze(tf.sqrt(tf.square(Drg) + tf.square(Drb) + tf.square(Dbg)))
    
    def get_config(self):
        return super(ColorConstancyLosss, self).get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class IlluminationSmoothnessLoss(tf.keras.losses.Loss):
    '''
    To preserve the monotonic relations between 
    neighboring pixels, the author of the paper added an illumination
    smoothness loss to each curve parameter map A.
    '''
    def __init__(self, **kwargs):
        super(IlluminationSmoothnessLoss, self).__init__(**kwargs)
    
    def __call__(self, alpha_map: tf.Tensor)-> tf.Tensor:
        '''
        Args:
            alpha_map: The curve parameter map tensor.
        Returns:
            The illumination smoothness loss.
        '''
        batch_size = tf.shape(alpha_map)[0]
        h_alpha_map = tf.shape(alpha_map)[1]
        w_alpha_map = tf.shape(alpha_map)[2]

        count_h = (tf.shape(alpha_map)[2] - 1) * tf.shape(alpha_map)[3]
        count_w = tf.shape(alpha_map)[2] * (tf.shape(alpha_map)[3] - 1)

        batch_size = tf.cast(batch_size, dtype=tf.float32)
        count_h = tf.cast(count_h, dtype=tf.float32)
        count_w = tf.cast(count_w, dtype=tf.float32)

        # Calculate Total Variation along height direction
        tv_h = tf.reduce_sum(tf.square((alpha_map[:, 1:, :, :] - alpha_map[:, : h_alpha_map - 1, :, :])))
        tv_h = tv_h /count_h

        # Calculate Total Variation along width direction
        tv_w = tf.reduce_sum(tf.square((alpha_map[:, :, 1:, :] - alpha_map[:, :, : w_alpha_map - 1, :])))
        tv_w = tv_w /count_w

        return 2*(tv_h + tv_w)/batch_size
    
    def get_config(self):
        return super(IlluminationSmoothnessLoss, self).get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == '__main__':
    # Test the SpatialConsistencyLoss
    img = tf.random.normal((1, 256, 256, 3))
    img2 = tf.random.normal((1, 256, 256, 3))
    loss = SpatialConsistencyLoss()
    tf.print(f'SpatialConsistencyLoss: {loss(img,img2)}\n')

    # Test the ExposureControlLoss
    img = tf.random.normal((1, 256, 256, 3))
    loss = ExposureControlLoss(e=0.6, patch_size=16)
    tf.print(f'ExposureControlLoss: {loss(img)}\n')

    # Test the ColorConstancyLoss
    img = tf.random.normal((1, 256, 256, 3))
    loss = ColorConstancyLosss()
    tf.print(f'ColorConstancyLoss: {loss(img)}\n')

    # Test the IlluminationSmoothnessLoss
    img = tf.random.normal((1, 256, 256, 3))
    loss = IlluminationSmoothnessLoss()
    tf.print(f'IlluminationSmoothnessLoss: {loss(img)}\n')
    