import numpy as np
import tensorflow as tf


def get_generator(input, training_phase=False):
    net = tf.layers.dense(input, 1024, kernel_initializer=tf.uniform_unit_scaling_initializer(1.43))
    net = tf.layers.batch_normalization(net, training=training_phase)
    net = tf.nn.relu(net)
    net = tf.layers.dense(net, 7 * 7 * 128,
                          kernel_initializer=tf.uniform_unit_scaling_initializer(1.43))
    net = tf.layers.batch_normalization(net, training=training_phase)
    net = tf.nn.relu(net)
    net = tf.reshape(net, [-1, 7, 7, 128])
    net = tf.layers.conv2d_transpose(net, 64, 4, strides=2, padding='SAME',
                                     kernel_initializer=tf.uniform_unit_scaling_initializer(1.43))
    net = tf.layers.batch_normalization(net, training=training_phase)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, 1, 4, strides=2, padding='SAME',
                                     kernel_initializer=tf.uniform_unit_scaling_initializer(1.43))
    net = tf.nn.sigmoid(net)
    return net


def get_discriminator(input, training_phase=False, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net = tf.layers.conv2d(input, 64, 4, strides=2, padding='SAME',
                           kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.43), name='d_conv_1')
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, 128, 4, strides=2, padding='SAME',
                           kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.43), name='d_conv_3')
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], padding='SAME', name='d_pool_2')
    shape = net.get_shape().as_list()
    dim = np.prod(shape[1:])
    net = tf.reshape(net, [-1, dim], name='d_reshape')
    net = tf.layers.dense(net, 1024, name='d_dense_1')
    net = tf.nn.relu(net)
    net = tf.layers.dense(net, 1, name='d_dense_2')
    return net, tf.nn.sigmoid(net)
