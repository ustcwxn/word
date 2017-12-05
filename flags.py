# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 01:30:47 2017

@author: lechao
"""
import tensorflow as tf
tf.app.flags.DEFINE_string('device','/cpu:0','use device')
tf.app.flags.DEFINE_integer('train_batch_size',10,'')
tf.app.flags.DEFINE_integer('train_iter',100000,'')
tf.app.flags.DEFINE_integer('valid_batch_size',128,'')
tf.app.flags.DEFINE_integer('valid_iter',28,'')
tf.app.flags.DEFINE_integer('display_gap',100,'show train loss per display_gap train iteration')
tf.app.flags.DEFINE_integer('valid_gap',4,'do validation per valid_gap train iteration')
tf.app.flags.DEFINE_string('train_label_list','/space2/lechao/MyWork/myself/train.txt','')
tf.app.flags.DEFINE_string('val_label_list','/space2/lechao/MyWork/myself/valid.txt','')
tf.app.flags.DEFINE_string('file_root_train', '/space2/lechao/MyWork/myself/train','')
tf.app.flags.DEFINE_string('file_root_valid','/space2/lechao/MyWork/myself/valid','')


FLAGS = tf.app.flags.FLAGS