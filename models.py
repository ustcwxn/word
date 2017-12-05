# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:11:42 2017

@author: lechao
"""

import tensorflow as tf
import numpy as np
import logging
from utils import *
from module import *
import flags

FLAGS = flags.FLAGS
class Config():
    def __init__(self):
        self.in_shape = [128,128,3] 
        
class ResNet(object):
    def __init__(self,name="model"):
        #self.config = config
        self.in_shape = [128,128,3]
        self.name = name
        #self.is_train = is_train
        #if tf.reset_default_graph()==None:
        self.build_graph()
    def build_graph(self):
        with tf.device(FLAGS.device):
            self.img_batch= tf.placeholder(tf.float32,shape=[None, self.in_shape[0], self.in_shape[1], self.in_shape[2]])
            self.img_labels = tf.placeholder(tf.int64, shape=[None, 1])
            self.is_train  = tf.placeholder(tf.bool)
            self.img_labels = tf.squeeze(self.img_labels)
            self.conv1 = Conv_bn_relu([7,7,self.in_shape[2],64],[1,2,2,1],name="conv_bn_relue1")
        
            #self.conv1 = self.img_batch
            self.conv1_out = self.conv1.apply(self.img_batch, self.is_train)
            self.pool1_out = non_global_pool(self.conv1_out)
            shape_list = self.pool1_out.get_shape().as_list()
            self.res_block = Resblock(shape_list[-1], [3,3], "res_block")
            self.res_out = self.res_block.apply(self.pool1_out, self.is_train)
            self.red = tf.reduce_mean(self.res_out, [1,2])
            shape_list = self.red.get_shape().as_list()
            self.w = norm_weight([shape_list[-1], 10])
            self.b = const_weight([10])
            self.logits = tf.matmul(self.red, self.w) + self.b
            self.label = tf.one_hot(self.img_labels, 10)
            
            self.preds = tf.one_hot(tf.arg_max(self.logits, 1),10)
            self.loss = tf.reduce_sum(self.preds * tf.nn.softmax(self.logits))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.preds, self.label), tf.float32))
            self.optimizer = adam_optimize(self.loss)
"""
if __name__ == "__main__":
    with tf.device(FLAGS.device):
        #rnn_data = tf.ones([10,5,20],dtype=tf.float32)
        #c2 = RNN(20,15,False,False,1)
        #b=c2.apply(rnn_data)
        a =tf.ones([1,4,4,20])
        res = ResNet("model")
        #aaa = c2.apply(a)
        #c2 = Res_unit_v2(20,20,name='resunit')  
        #aaa = c2.apply(a,tf.constant(True))
        #c2 = Conv_bn_relu([3,3,2,2],name="convmodule1")
        #aaa = c2.apply(a)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(tf.initialize_all_variables())
        feed_dict={res.img_batch:np.ones([10,128,128,3]),res.img_labels:np.zeros([10,1]),res.is_train:True}
        print sess.run([res.loss],feed_dict=feed_dict) 
"""                                                                                                                