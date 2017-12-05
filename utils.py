# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:09:23 2017

@author: lechao
"""

import numpy as np
import tensorflow as tf
import logging
from tensorflow.python.training.moving_averages import assign_moving_average
def const_weight(shape, value=0.0):
    return tf.Variable(tf.constant(value, shape=shape, dtype=tf.float32))
    
def norm_weight(shape, mean=0.0, std=1.0):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=std,dtype=tf.float32))


def global_pool(x, axis,name="global_pool"):
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        axis = list(axis)
        for i in axis:
            if i not in shape:
                logging.warn("global pool:  axis:%d is not int tensor shape\n"%(i))
                return
        return tf.reduce_mean(x,axis)

def non_global_pool(x, ksize = [1,3,3,1], strides = [1,2,2,1], type=0, padding='SAME'):
    #max pooling
    if type not in [0,1]:
        logging.warn("wrong pooling type!\n")
        return 
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding) if type==0 else tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding=padding)
    
"""   
def batch_normalization(input_layer, is_train, eps=1e-5, decay=0.9, affine=True, name="batch_norm"):
    with tf.variable_scope(name) as scope:
        shape_list = input_layer.get_shape()
        params_shape = shape_list[-1:]
        #scope.reuse_variables()
        moving_mean =tf.get_variable('mean',params_shape,
                                     initializer=tf.zeros_initializer,
                                     trainable=False)
        moving_variance =tf.get_variable('var',params_shape,
                                         initializer=tf.zeros_initializer,
                                         trainable=False)
        def mean_var_with_update():
            mean,variance =tf.nn.moments(input_layer,list(range(len(shape_list)-1)),name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean,mean,decay),assign_moving_average(moving_variance,variance,decay)]):
                return tf.identity(mean),tf.identity(variance)
            
        mean,variance = tf.cond(is_train,mean_var_with_update,lambda:(moving_mean,moving_variance)) 
        if affine:
            beta = tf.get_variable('beta',params_shape,initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma',params_shape,initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(input_layer,mean,variance,beta,gamma,eps)
        else:
            x=tf.nn.batch_normalization(x,mean,variance,None,None,eps)
        return x   
            
def conv_bn_relu_layer(input_layer, filter_shape, stride, is_train=True ,name= "conv_bn_relu"):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    ''' 
    #out_channel = filter_shape[-1]
    #kernel = tf.Variable(shape =filter_shape,name = 'conv')
    #filter = create_variables(name='conv', shape=kernel)
    conv_layer = conv2d(input_layer, filter_shape, strides= [1,stride,stride,1], padding='SAME',name = name+"_conv2d")
    bn_layer = batch_normalization(conv_layer, tf.constant(True,dtype=bool), name = name+"batch_norm")
    output = tf.nn.relu(bn_layer)
    return output
    
def bn_relu_conv_layer(input_layer, filter_shape, stride,is_train=True):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    pass
    #in_channel = input_layer.get_shape().as_list()[-1]
    #bn_layer = batch_normalization(input_layer, is_train)
    #relu_layer = tf.nn.relu(bn_layer)
    #filter = create_variables(name='conv', shape=filter_shape)
    #conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    #return conv_layer  
    
def residual_unit_v2(input_layer, output_channel,is_train=True, name = "residual_unit"):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        #stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        #stride = 1
    #else:
    #    raise ValueError('Output and input channel does not match in residual blocks!!!')
    #with tf.variable_scope('conv1_in_block'):
    conv1 = conv_bn_relu_layer(input_layer, [1, 1, input_channel, output_channel // 4], 1,is_train, name=name+"conv1")
    #with tf.variable_scope('conv2_in_block'):
    conv2 = conv_bn_relu_layer(conv1, [3, 3, output_channel//4, output_channel // 4], 1, is_train, name=name+"conv2")
    #with tf.variable_scope('conv3_in_block'):
    conv3 = conv_bn_relu_layer(conv2, [1, 1, output_channel//4, output_channel], 1, is_train, name=name+"conv3")
    if increase_dim is True:
        transform_input = conv_bn_relu_layer(input_layer, [1, 1, input_channel,output_channel],1, is_train, name=name+"conv4")
    else:
        transform_input = input_layer
    output = conv3 + transform_input
    return output
    
def ResBlock(input_layer, block_num_list, type_list, is_train, name = "res_block"):
    if len(block_num_list) != len(type_list):
        raise ValueError('block_num_list length is not equal to type_list length!!')
    shape_list = input_layer.get_shape().as_list()
    input_channel = shape_list[-1]
    output_layer = input_layer
    for sel,[num, type_id] in enumerate(zip(block_num_list,type_list)):
        for i in xrange(num):
            output_layer = residual_unit_v2(output_layer, input_channel, is_train, name=name+"_"+str(sel)+"_"+str(i))
        input_channel *= 2
    return output_layer 
 """
def adam_optimize(cost):
    optimzer = tf.train.AdamOptimizer()
    train_vars = tf.trainable_variables()
    l2 = sum([tf.reduce_sum(x**2)for x in train_vars])
    train_cost = cost + l2*1e-6
    grads_and_vars = optimzer.compute_gradients(train_cost)
    grads, vars_ = zip(*grads_and_vars)
    grads,_ = tf.clip_by_global_norm(grads, 5.0)
    return optimzer.apply_gradients(zip(grads, vars_))