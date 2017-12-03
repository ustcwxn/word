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
    def __init__(self,stream_train=None,stream_val=None,stream_test=None):
        self.stream_train = stream_train
        self.stream_val = stream_val
        self.stream_test = stream_test
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        self.epoch = 1
        self.build_graph()
        self.stream_train.num = 10734
    def __del__(self):
        self.sess.close()
    def train(self):
        
        with tf.device('/gpu:1'):    
            self.sess.run(tf.initialize_all_variables())
            file_queue = tf.train.string_input_producer(["train_zhiwu_record"])
            img,label = self.stream_train.read_record(file_queue) 
            img_batch,label_batch = tf.train.shuffle_batch([img,label],
                                                                           batch_size =100,
                                                                           capacity = 20000,
                                                                           min_after_dequeue=10000,
                                                                           num_threads=4
                                                                           ) 
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = self.sess, coord=coord) 
          
            step =0
            try:
                while not coord.should_stop():             
                    step+=1
                    imgs,labels=self.sess.run([img_batch,label_batch])
                    feed_dict = {self.img_batch:imgs,self.img_labels:labels,self.is_train:True}
                    loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
                    
                    
                    
                    print 'step=%d  loss=%.6f\n'%(step,loss)
                    if step>100:
                        break
            except tf.errors.OutOfRangeError:
                print 'input stream was exausted  by wxn \n'
            finally:
                coord.request_stop()
            
            coord.request_stop()
            coord.join(threads)
                    #print self.sess.run(tf.shape(img_batch))
                    #break
                    #feed_dict = {self.img_batch:img_batch,self.img_labels:label_batch,self.is_train:True}
                    #loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
                    #if iter_num % 10 ==0:
                        #logging.info('epoch :%d iter : %d  train loss:%.6f \n'%(epoch_num,iter_num,loss))
                       # break
            #except tf.errors.OutOfRangeError:
                #print 'Done traing -- input stream was exausted\n'
            #finally:
                #coord.request_stop()
                    #if iter_num % 1000==0:
                    #   validation()
            #coord.request_stop()
            #coord.join(threads)
        self.sess.close()
            
            
     
    def build_graph(self):
        self.core = ResNet()
        self.loss = self.core.loss
        self.is_train = self.core.is_train
        self.optimizer = self.core.optimizer
        self.img_batch = self.core.img_batch
        self.img_labels = self.core.img_labels
         
    def validation(self,stream_val):
        batch_size = 30
        iters = 100

train_label_list = '/space2/lechao/MyWork/myself/train.txt'
val_label_list = '/space2/lechao/MyWork/myself/valid.txt'
#test_label_list = '../11.lstm-test/label-test.txt'
file_root_train = '/space2/lechao/MyWork/myself/train'
file_root_valid = '/space2/lechao/MyWork/myself/valid'


stream_train = Stream(train_label_list,file_root_train,"train_zhiwu_record")
stream_val = Stream(val_label_list,file_root_valid,"valid_zhiwu_record")
predictor = Predictor(stream_train, stream_val)
predictor.train()