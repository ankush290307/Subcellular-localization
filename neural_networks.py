#!/usr/bin/env python

#__author__ "Ankush Sharma"

import tensorflow as tf
import numpy as np
import tflearn.activations as tf_acts
leaky_relu = tf_acts.leaky_relu
alpha = 0.1


def get_batch(tensor, n=100):
#Gets a minibatch from a tensor

    #Takes a tensor of shape t = [[[seq_1][lab_1]], ..., [[seq_n][lab_n]]] and
   # randomly takes n samples, returning a tensor x = [[seq_1], ..., [seq_n]]
   # and a tensor y = [[lab_1], ..., [lab_n]].
    
    idxs = np.random.choice(len(tensor), n, replace=False)
    x = [tensor[i][0] for i in idxs]
    y = [tensor[i][1] for i in idxs]

    return x, y

#Generating weight variables
def weight_variable(shape, name="W"):
    

 #Provides a tensor of weight variables obtained from a truncated normal distribution with mean=0 and std=0.1. All values in range[-0.1,0.1]
    

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

# defining bias provides a tensor of bias variables with value 0.1
def bias_variable(shape, name="B"):
   
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

#Full connected Neural network with adding the previously defined biases and weights Computes a fully connected layer when provided with an input tensor and returns an output tensor. Input and output channels must be specified.By default, the output uses a leaky ReLu activation function.
def fc_layer(input_tensor, input_dim, output_dim, name="fc", relu=True):
    
    with tf.name_scope(name):
        w = weight_variable([input_dim, output_dim])
        b = bias_variable([output_dim])
        out = tf.matmul(input_tensor, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)

        if relu:
            return leaky_relu(out, alpha, name=name)
        else:
            return out

#Convolutional Neural network layer 
def conv_layer(input_tensor, width, heigth, in_channels, out_channels,
               name="conv", relu=True)

    with tf.name_scope(name):

        w = tf.get_variable(name+"-weights",
                            [width, heigth, in_channels, out_channels])

        b = bias_variable([out_channels])
        conv = tf.nn.conv2d(input_tensor, w,
                            strides=[1, 1, 1, 1], padding="SAME")

        if not relu:
            conv_norelu = conv + b

            return conv_norelu

        conv_relu = leaky_relu(conv + b, alpha, name=name)

        tf.summary.histogram("filter_weights", w)
        tf.summary.histogram("biases", b)

        return conv_relu

#LSTM is copied from github code not sure if itwill work
def LSTM(x, num_units, output, name="LSTM", fb=1.0):
    with tf.variable_scope(name):

        w = weight_variable([num_units, output])
        b = bias_variable([output])
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=fb)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell,
                                                    x, dtype=tf.float32)

        return tf.matmul(outputs[-1], w) + b

#Performs a max pooling on the input tensor
def max_pool_layer(conv_tensor, width, height, channels, padding="SAME",
                   name="max_pool"):

    out = tf.nn.max_pool(conv_tensor, ksize=[1, width, 1, 1],
                         strides=[1, width, 1, 1], padding=padding,
                         name=name)

    return out


def trainable_input(shape, name="trainable_x", aa_vec_len=20):
    initial = tf.constant(1 / aa_vec_len, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name)