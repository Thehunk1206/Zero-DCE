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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2 as cv

from utils import get_model, post_enhance_iteration


def process_frame(in_frame: np.ndarray, img_h:int = 160, img_w:int = 160) -> tf.Tensor:
    assert isinstance(in_frame, np.ndarray)
    
    frame = tf.convert_to_tensor(in_frame, dtype=tf.float32)
    frame = tf.image.resize(frame, (img_h, img_w), method=tf.image.ResizeMethod.BICUBIC)
    frame = frame / 255.0
    frame = tf.expand_dims(frame, axis=0)

    return frame


def run_video_inference(model_path:str, video_path:str="__camera__" , img_h:int = 160, img_w = 160, downsample_factor:int = 1.0):
    assert 0.1 <= downsample_factor <= 1.0 

    zero_dce_model =  get_model(model_path)

    # Read video
    if video_path == "__camera__":
        cap = cv.VideoCapture(2)
    else:
        assert os.path.exists(video_path), "Video file not found"
        cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()
    else:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(5)
        print(f'Frame size: {frame_width}x{frame_height}')
        print('Frames per second : ', fps, 'FPS')
    
    # constants for fps calculation
    font = cv.FONT_HERSHEY_SIMPLEX
    PREV_FRAME_TIME = 0
    NEW_TIME_FRAME = 0
    
    # Creating a Trackbar for post enhancing iteration value change.
    cv.namedWindow('Set Post Enhancing iteration')
    cv.resizeWindow('Set Post Enhancing iteration', 300, 100)
    cv.createTrackbar('iteration', 'Set Post Enhancing iteration', 1, 9, lambda x: None)
    cv.setTrackbarPos('iteration', 'Set Post Enhancing iteration', 6)

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            frame = cv.flip(frame,1)
            frame = cv.resize(frame, (int(frame_width*downsample_factor), int(frame_height*downsample_factor)), interpolation = cv.INTER_CUBIC)

            input_frame = process_frame(frame, img_h=img_h, img_w=img_w)
            frame = tf.convert_to_tensor((frame/255.0), dtype=tf.float32)
            _, a_maps = zero_dce_model(input_frame)

            a_maps = tf.squeeze(a_maps, axis=0)
            a_maps = cv.GaussianBlur(a_maps.numpy(), (5,5), 0)
            a_maps = tf.cast(a_maps, dtype=tf.float32)
            post_enh_iteration = cv.getTrackbarPos('iteration', 'Set Post Enhancing iteration')
            enhanced_frame = post_enhance_iteration(frame, a_maps, iteration=post_enh_iteration)

            # get numpy array from tensor
            enhanced_frame = enhanced_frame.numpy()
            frame = (frame.numpy()*255.0).astype(np.uint8)

            # Calculating and displaying FPS
            NEW_TIME_FRAME = time()
            if (NEW_TIME_FRAME - PREV_FRAME_TIME) > 0:
                fps = 1.0 / (NEW_TIME_FRAME - PREV_FRAME_TIME)
                PREV_FRAME_TIME = NEW_TIME_FRAME
            else:   
                fps = 0.0
            show_verbose = f'enhanced_output FPS: {int(fps)} post_enhancing_iteration: {post_enh_iteration}'
            cv.putText(enhanced_frame, show_verbose, (7,20), font, 0.5, (0,0,255), 1, cv.LINE_AA)
            cv.putText(frame, "low_light_input", (7,20), font, 0.5, (0,0,255), 1, cv.LINE_AA)

            # Combine both frames
            combined_frame = np.concatenate((frame, enhanced_frame), axis=1)

            cv.imshow('frame', combined_frame)

        else:
            break

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

# main function that created command line arguments and runs the video inference
def main():
    parser = argparse.ArgumentParser(description='Zero DCE model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model file')
    parser.add_argument('--video_path', type=str, default='__camera__',help='Path to the video file. If not given the camera will be used')
    parser.add_argument('--img_h', type=int, default=160, help='Image height')
    parser.add_argument('--img_w', type=int, default=160, help='Image width')
    parser.add_argument('--downsample_factor', type=float, default=1.0, help='Downsample factor')
    args = parser.parse_args()

    run_video_inference(
        model_path=args.model_path,
        video_path=args.video_path,
        img_h=args.img_h,
        img_w=args.img_w,
        downsample_factor=args.downsample_factor
    )



if __name__ == "__main__":
    main()
