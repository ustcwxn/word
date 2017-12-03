# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:09:56 2017

@author: lechao
"""
import tensorflow as tf
import numpy as np
import logging
from stream import *
from models import *

logging.basicConfig(level=logging.DEBUG, 
                    format = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt = '%a, %d %b %Y %H:%M:%S',
                    filename = 'train.log',
                    filemode = 'w')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)  

class Predictor():
    def __init__(self,stream_train,stream_val,stream_test):
        self.stream_train = stream_train
        self.stream_val = stream_val
        #self.strea_test = steam_test
        self.sess = tf.Session()
        self.epoch = 20
        build_graph()
    def __del__(self):
        self.sess.close()
    def train(self):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = self.sess,coord=coord)
        img,label = stream_train.read_record()
        img_batch,label_batch = tf.train.shuffle_batch([img,label],
                                                           batch_size =30,
                                                           capacity = 3100,
                                                           min_after_dequeue=3000,
                                                           num_threads=3)   
        init = tf.initialize_all_variables()     
        self.sess.run(init)
        iter_num = 0
        
        for epoch_num in xrange(self.epoch):
            for i in xrange(self.stream_train.num/30):
                iter_num+=1
                img_batch,label_batch = self.sess.run([img_batch,label_batch])
                feed_dict = {self.img_batch:img_batch,self.img_labels:label_batch,self.is_train:True}
                loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
                logging.info('epoch :%d iter : %d  train loss:%.6f \n'%(epoch_num,iter_num,loss))
                #if iter_num % 1000==0:
                 #   validation()
        coord.request_stop()
        coord.join(threads)
        
    def build_graph(self):
        self.core = ResNet()
        self.loss = self.core.loss
        self.optimizer = self.core.optimizer
        self.img_batch = self.core.img_batck
        self.img_labels = self.core.img_labels
         
    def validation(self,stream_val):
        batch_size = 30
        iters = 100
         
        