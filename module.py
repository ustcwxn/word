# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:29:07 2017

@author: lechao
"""
import numpy as np
import tensorflow as tf
import logging
from utils import *
from tensorflow.python.training.moving_averages import assign_moving_average

#from tensorflow.contrib.rnn import BasicLSTMCell
#from tensorflow.contrib.rnn import GRUCell
#from tensorflow.contrib.rnn import LSTMStateTuple
#from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell

class DenseLayer(object):
    def __init__(self, dim_in, dim_out, name="dense"):            
        self.w = norm_weight([dim_in,dim_out])
        self.b = const_weight([dim_out])
    def apply(self, input_tensor):
        self.shape_list = tf.shape(input_tensor).as_list()
        input_reshaped = tf.reshape(input_tensor, shape= [-1, tf.shape(input_tensor)[-1]])
        out = tf.matmul(input_reshaped, self.w) + self.b
        out_shaped = tf.reshape(out, tf.concat([tf.shape(input_tensor)[:-1],tf.shape(self.b)],axis=0))
        return out_shaped
        
class Conv2d(object):
    def __init__(self, kernel_shape, stride = [1,1,1,1], padding ='SAME', name="conv2d"):
        self.name = name
        with tf.device('/cpu:0'):
            with tf.variable_scope(self.name):
                self.kernel_shape = kernel_shape
                self.padding = padding
                self.stride = stride       
                self.W = norm_weight(self.kernel_shape, 0.0, 0.118)
                self.b = const_weight(self.kernel_shape[-1:], 0.0)
    def apply(self, x):
        #tf.assert_equal(tf.shape(x)[3], self.kernel_shape[2])
        return tf.nn.bias_add(tf.nn.conv2d(x, self.W, strides=self.stride, padding=self.padding),self.b)
        
class Conv_bn_relu(object):
    def __init__(self, kernel_shape, stride=[1,1,1,1], padding='SAME', name='bn_conv_relu'):
        self.name = name
        self.kernel_shape = kernel_shape
        with tf.device('/cpu:0'):
            with tf.variable_scope(self.name):
                self.conv2d = Conv2d(self.kernel_shape, stride, padding, "conv2d")
                with tf.variable_scope("bn"):
                    self.beta = tf.Variable(tf.zeros(self.kernel_shape[-1:], dtype=tf.float32),name="beta")
                    self.gamma = tf.Variable(tf.ones(self.kernel_shape[-1:], dtype=tf.float32),name="gamma")
                    self.moving_mean = tf.Variable(tf.zeros(self.kernel_shape[-1:], dtype=tf.float32),name="m_mean",trainable=False)
                    self.moving_variance = tf.Variable(tf.zeros(self.kernel_shape[-1:], dtype=tf.float32),name="m_var",trainable=False)
    def apply(self, x, is_train=True, eps=1e-5, decay=0.9):
        with tf.variable_scope(self.name):
            self.x_conv = self.conv2d.apply(x)
            x_conv_shape = self.x_conv.get_shape().as_list()
            def mean_var_with_update():              
                mean,variance = tf.nn.moments(self.x_conv,list(range(len(x_conv_shape)-1)),name='moments')
                with tf.control_dependencies([assign_moving_average(self.moving_mean,mean,decay),assign_moving_average(self.moving_variance,variance,decay)]):
                    return tf.identity(mean),tf.identity(variance)
            mean,variance = tf.cond(tf.convert_to_tensor(is_train),mean_var_with_update,lambda:(self.moving_mean,self.moving_variance)) 
            return tf.nn.relu(tf.nn.batch_normalization(self.x_conv, mean, variance, self.beta, self.gamma, eps))
class Res_unit_v2(object):
    def __init__(self, in_channel, out_channel ,name="res_unit_v2"):
        self.name = name
        self.in_channel = in_channel
        self.out_chnnel = out_channel
        if out_channel!=in_channel and out_channel!=in_channel*2 and out_channel//4 <=0:
            raise ValueError("res unit input output channel numbers are ilegal or input channel is too small\n")
        self.increase_dim = True if in_channel*2 == out_channel else False
        with tf.device('/cpu:0'):
            with tf.variable_scope(self.name):
                self.conv1 = Conv_bn_relu([1,1,in_channel,out_channel//4],name="convmodule1")
                self.conv2 = Conv_bn_relu([3,3,out_channel//4,out_channel//4],name="convmodule2")
                self.conv3 = Conv_bn_relu([1,1,out_channel//4,out_channel],name="convmodule3")
                self.conv4 = Conv_bn_relu([1,1,in_channel,out_channel],name="convmodule4") if self.increase_dim==True else None
    def apply(self, x, is_train=True):
        x_shape = x.get_shape().as_list()
        if x_shape[-1] != self.in_channel:
            raise ValueError("res unit input is not match: real input channel = %d, resunit input channel =%d \n"%(x_shape[-1],self.in_channel))
        with tf.variable_scope(self.name):
            conv1 = self.conv1.apply(x, is_train=is_train)
            conv2 = self.conv2.apply(conv1, is_train=is_train)
            conv3 = self.conv3.apply(conv2, is_train=is_train)
            return conv3 if self.increase_dim==False else conv3+self.conv4.apply(x)

class Resblock(object):
    def __init__(self, in_channel, block_num_list, name = "res_block"):
        self.name = name
        self.res_unit = []
        self.in_channel = in_channel
        self.block_num_list = block_num_list
        out_channel = in_channel
        with tf.device('/cpu:0'):
            with tf.variable_scope(self.name):
                for i,num in enumerate(self.block_num_list):
                    for j in xrange(num):
                        self.res_unit.append(Res_unit_v2(in_channel, out_channel, name="res_unit"+str(i)+"_"+str(j)))
                        in_channel = out_channel
                    out_channel = out_channel *2                   
    def apply(self, x, is_train=True):
        x_shape = x.get_shape().as_list()
        output = x
        if x_shape[-1]!=self.in_channel:
            raise ValueError("resblock input channel is not match the net input design\n")
        flag = 0
        for i,num in enumerate(self.block_num_list):
            for j in xrange(num):
                output = self.res_unit[flag].apply(output, is_train)
                flag = flag +1
        return output  
class RNN(object):
    def __init__(self, dim_in, dim_out, with_init, with_atten, num_layer, reuse=False, keep_prob=1.0, name="rnn"):
        self.name = name
        self.type = "LSTM"
        self.reuse = reuse
        self.dim_in = dim_in
        with tf.variable_scope(self.name):
            self.with_init = with_init
            self.with_atten = with_atten
            if with_init:
                self.W_init = norm_weight([dim_in,dim_out], std=0.1)
                self.b_init = const_weight([dim_out], 0.0)
            if with_atten:
                self.W_atten = norm_weight([dim_out * num_layer, dim_in])
            f_lstm_cell = lambda d:BasicLSTMCell(d, forget_bias =1.0, state_is_tuple=True)
            f_gen_cell = f_lstm_cell(dim_out) if self.type=='LSTM'else GRUCell(dim_out)
            cells = [f_gen_cell for i in num_layer]
            cells_drop = [DropoutWrapper(x,output_keep_prob=keep_prob) for x in cells]
            self.multi_lstm = MultiRNNCell(cells, state_is_tuple=True)
            self.multi_lstm_drop = MultiRNNCell(cells_drop, state_is_tuple=True)
    def apply(self, x, init_ctx=None, seq_mode=False, is_train=True):
        shape_list = x.get_shape().as_list()
        if len(shape_list)!=3:
            raise ValueError("RNN input tensor dim was not 3!\n")
        with tf.variable_scope(self.name):
            rnn = self.multi_lstm_drop if is_train else self.multi_lstm
            outputs= []
            state = rnn.zero_state(shape_list[0], tf.float32)
            if self.with_init and init_ctx!=None:
                state = [self.init_state(init_ctx)] * len(state)
            for time_step in range(shape_list[1]):
                if self.reuse or not is_train or time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cur_ctx = self.attention(x, state) if self.with_atten else x[: ,time_step, :]
                output ,state = rnn(cur_ctx ,state)
                outputs.append(output)
            return outputs if seq_mode else outputs[-1]
    def init_state(self,  init_ctx):
        with tf.variable_scope(self.name):
            init = tf.matmul(init_ctx, self.W_init) + self.b_init
            return LSTMStateTuple(init, tf.nn.tanh(init)) if self.type == "LSTM" else init
    def attention(self, x, state):
        state = tf.convert_to_tensor(state)
        state = state[:, 1, :, :] if self.type == "LSTM" else state
        reshaped = tf.reshape(tf.transpose(state, perm=[1, 0, 2]), shape=[tf.shape(state)[-2], -1])
        scores = tf.reduce_sum(tf.expand_dims(tf.matmul(reshaped, self.W_atten), axis =1)*x,axis=-1)
        scores = tf.exp(scores)
        probs = scores / tf.reduce_sum(scores, axis=-1, keep_dims=True)
        return tf.reduce_sum(x * tf.expand_dims(probs, axis=-1), axis=1)
if __name__ == "__main__":
    with tf.device('/cpu:0'):
        rnn_data = tf.ones([10,5,20],dtype=tf.float32)
        c2 = RNN(20,15,False,False,1)
        b=c2.apply(rnn_data)
        #a =tf.ones([1,4,4,20])
        #c2 = Resblock(20,[2,2],name='rb')
        #aaa = c2.apply(a)
        #c2 = Res_unit_v2(20,20,name='resunit')  
        #aaa = c2.apply(a,tf.constant(True))
        #c2 = Conv_bn_relu([3,3,2,2],name="convmodule1")
        #aaa = c2.apply(a)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print sess.run(aaa)