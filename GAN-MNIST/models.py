import numpy as np
import tensorflow as tf


def get_generator(input):
    net = tf.layers.dense(input, 7 * 7 * 256, activation=tf.nn.relu,
                          kernel_initializer=tf.uniform_unit_scaling_initializer(1.43))
    net = tf.reshape(net, [-1, 7, 7, 256])
    net = tf.layers.conv2d_transpose(net, 128, [2, 2], strides=[2, 2], activation=tf.nn.relu,
                                     kernel_initializer=tf.uniform_unit_scaling_initializer(1.43))
    net = tf.layers.conv2d_transpose(net, 64, [2, 2], strides=[2, 2], activation=tf.nn.relu,
                                     kernel_initializer=tf.uniform_unit_scaling_initializer(1.43))
    net = tf.layers.conv2d(net, 1, [3, 3], padding='same', activation=tf.nn.sigmoid)
    return net


def get_discriminator(input, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net = tf.layers.conv2d(input, 32, [5, 5], activation=tf.nn.relu, padding='same',
                           kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.43), name='d_conv_1')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], padding='same', name='d_pool_1')
    net = tf.layers.conv2d(net, 64, [5, 5], activation=tf.nn.relu, padding='same',
                           kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.43), name='d_conv_3')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], padding='same', name='d_pool_2')
    shape = net.get_shape()
    dim = np.prod(shape[1:])
    net = tf.reshape(net, [-1, dim.value], name='d_reshape')
    net = tf.layers.dense(net, 1024, activation=tf.nn.sigmoid, name='d_dense_1')
    net = tf.layers.dense(net, 1, name='d_dense_2')
    return net, tf.nn.sigmoid(net)
