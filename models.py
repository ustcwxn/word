# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:11:42 2017

@author: lechao
"""

import tensorflow as tf
import numpy as np
import logging
from utils import *

class Config():
    def __init__(self):
        self.in_shape = [128,128,3] 
        
class ResNet:
    def __init__(self,name="model"):
        #self.config = config
        self.in_shape = [128,128,3]
        self.name = name
        #self.is_train = is_train
        #if tf.reset_default_graph()==None:
        self.build_graph()
    def build_graph(self):
        with tf.device('/gpu:1'):
            self.img_batch= tf.placeholder(tf.float32,shape=[None, self.in_shape[0], self.in_shape[1], self.in_shape[2]])
            self.img_labels = tf.placeholder(tf.int64, shape=[None, 1])
            self.is_train  = tf.placeholder(tf.bool)
            block_num = [1,1,1,1]
            type_list = [2,2,2,2]
            #self.conv1 = self.img_batch
            self.conv1 = conv_bn_relu_layer(self.img_batch, [7,7,self.in_shape[2],64], 2, self.is_train, name="c_b_r")
            self.pool1 = non_global_pool(self.conv1)
            #self.res_out = ResBlock(self.pool1, block_num, type_list, self.is_train, name="res_block1")
            self.red = tf.reduce_mean(self.pool1, [1,2])
            shape_list = self.red.get_shape().as_list()
            self.w = norm_weight([shape_list[-1], 10])
            self.b = const_weight([10])
            self.logits = tf.matmul(self.red, self.w) + self.b
            self.label = tf.one_hot(tf.squeeze(self.img_labels), 10)
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.label))
            self.optimizer = adam_optimize(self.loss)
                                                                                                                  