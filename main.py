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
train_label_list = '/space2/lechao/MyWork/myself/train.txt'
val_label_list = '/space2/lechao/MyWork/myself/valid.txt'
#test_label_list = '../11.lstm-test/label-test.txt'
file_root_train = '/space2/lechao/MyWork/myself/train'
file_root_valid = '/space2/lechao/MyWork/myself/valid'

if __name__ == '__main__':
    stream_train = Stream(train_label_list,file_root_train,"train_zhiwu_record")
    stream_val = Stream(val_label_list,file_root_valid,"valid_zhiwu_record")
    predictor = Predictor(stream_train, stream_val)
    predictor.train()