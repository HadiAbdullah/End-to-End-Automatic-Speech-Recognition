# Used two sources for building this class:
# ctc_lambda_func and Lambda(ctc_lambda_func...) shamelessly coped from https://colab.research.google.com/drive/1CdB9rvImJCAl_U9yYVD6HqMFWup_RzpG#scrollTo=HBLwtxF35f0c
# get_length Shamelessly copied from https://github.com/rolczynski/Automatic-Speech-Recognition/blob/master/automatic_speech_recognition/pipeline/ctc_pipeline.py


import os
import sys
import pathlib

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse


def ctc_lambda_func(args):
    '''
    Setup the CTC loss as function.
    y_pred: The logits output from the model. Shape [batch_sz, times_steps, number_characters]
    labels: The tokenized transcription. Shape [batch_sz, label_length]
    label_length: The length of the transcription. Shape [batch_sz, 1]
    '''
    y_pred, labels, label_length = args
    
    def get_length(tensor):
        '''
        Returns the length of a tensor
        Reference:
        "Automatic-Speech-Recognition"
        (https://github.com/rolczynski/Automatic-Speech-Recognition/blob/master/automatic_speech_recognition/pipeline/ctc_pipeline.py)
        '''
        lengths = tf.math.reduce_sum(tf.ones_like(tensor), 1)
        lengths = tf.expand_dims(lengths, -1)
        return tf.cast(lengths, tf.int32)

    # extracts the number of time steps for the batch
    input_length = get_length(tf.math.reduce_max(y_pred, 2))

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_ctc_layer(logits, y_true, y_true_length):
    '''
    For a given set of input layers, returns the CTC layer.
    logits: The logits output from the model. Shape [batch_sz, times_steps, number_characters]
    y_true: The tokenized transcription. Shape [batch_sz, label_length]
    y_true_length: The length of the transcription. Shape [batch_sz, 1]
    '''
    
    loss = Lambda(ctc_lambda_func, output_shape=(None,), name='ctc')([logits, y_true, y_true_length])

    return loss