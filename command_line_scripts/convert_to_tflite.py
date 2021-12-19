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
from tqdm import tqdm
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

def run_performance_testing(model_content: bytes, data_points:int=500, num_threads:int = 4):
    '''
    Run Tflite runtime performance testing on desktop CPU
    args:
        model_content:bytes, Tflite model content
        data_points:int, Number of data points to test
        num_threads:int, Number of threads to use
    '''
    interpreter = tf.lite.Interpreter(model_content=model_content, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    tf.print(f'Input shape: {input_shape}')
    input_shape = [data_points, *input_shape]

    input_data = tf.random.normal(shape=input_shape, dtype=tf.float32)
    input_data = tf.unstack(input_data, axis=0)

    times = []
    for data in tqdm(input_data, desc="Running Tflite runtime performance testing", colour='green'):
        interpreter.set_tensor(input_details[0]['index'], data.numpy())
        start = time()
        interpreter.invoke()
        end = time()
        times.append(end-start)
    avg_time = sum(times[3:])/len(times[3:])

    tf.print(f'\nAverage time taken to run {data_points} inferences: {round(avg_time, 2)*1000} ms')


def convert_saved_model_to_tflite(model_path:str, output_path:str, data_points:int = 200, num_threads:int = 4, quantize:bool = False)->None:
    """
    Convert saved model to tflite
    args:
        model_path:str, Path to saved model
        output_path:str, Path to output tflite model
        data_points:int, Number of data points to test
        num_threads:int, Number of threads to use
        quantize:bool, Quantize model
    """
    assert os.path.exists(model_path), "Model path does not exist"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model_name = model_path.split('/')[-1]
    
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        model_name = model_name + "_quantized"
    tflite_model = converter.convert()
    
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model, gpu_compatibility=True) # Analyze the tflite model

    tf.print(f'\nRunning Tflite runtime performance testing on desktop CPU\n')
    run_performance_testing(model_content=tflite_model, data_points=100) # Run Tflite runtime performance testing 

    file = open(f'{output_path}{model_name}.tflite', "wb").write(tflite_model)
    tf.print(f'\nModel converted to {output_path}{model_name}.tflite\n Model size: {file} bytes')
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    tf.print(interpreter.get_input_details()[0]['dtype'])
    tf.print(interpreter.get_input_details()[0]['shape'])
    tf.print(interpreter.get_output_details()[0]['dtype'])
    tf.print(interpreter.get_output_details()[0]['shape'])
    tf.print(interpreter.get_output_details()[1]['dtype'])
    tf.print(interpreter.get_output_details()[1]['shape'])


def main():
    parser = argparse.ArgumentParser(description='Convert saved model to tflite')
    parser.add_argument('--model_path', required=True, type=str, help='Path to saved model')
    parser.add_argument('--output_path', type=str, default='TFLITE_models/', help='Path to output tflite model')
    parser.add_argument('--data_points', type=int, default=200, help='Number of data points to test')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use')
    parser.add_argument('--quantize', type=int, default=0, choices=[0,1], help='Quantize model, 0: Do not Quantize, 1: Quantize')
    args = parser.parse_args()
    convert_saved_model_to_tflite(
        args.model_path, 
        args.output_path,
        args.data_points,
        args.num_threads,
        bool(args.quantize)
    )

if __name__ == "__main__":
    main()