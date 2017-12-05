# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:07:39 2017
@author: lechao
"""
import tensorflow as tf
import numpy as np
import logging
from stream import *
from predictor import *
import flags
FLAGS = flags.FLAGS
train_label_list = FLAGS.train_label_list
val_label_list = FLAGS.val_label_list
#test_label_list = '../11.lstm-test/label-test.txt'
file_root_train = FLAGS.file_root_train
file_root_valid = FLAGS.file_root_valid

def main():
    stream_train = Stream(train_label_list,file_root_train,"train_zhiwu_record")
    stream_val = Stream(val_label_list,file_root_valid,"valid_zhiwu_record")
    predictor = Predictor(stream_train, stream_val)
    predictor.train()
if __name__ =="__main__":
    tf.app.run(main)