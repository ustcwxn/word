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


import flags
FLAGS = flags.FLAGS

class Predictor():
    def __init__(self,stream_train=None,stream_val=None,stream_test=None):
        self.stream_train = stream_train
        self.stream_val = stream_val
        self.stream_test = stream_test
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        self.train_batch = FLAGS.train_batch_size
        self.val_batch = FLAGS.valid_batch_size
        self.train_iter = FLAGS.train_iter
        self.val_iter  = FLAGS.valid_iter
        self.build_graph()
    def __del__(self):
        self.sess.close()
    def train(self):
        
        with tf.device(FLAGS.device):    
            self.sess.run(tf.initialize_all_variables())
            file_queue = tf.train.string_input_producer(["train_zhiwu_record"])
            img,label = self.stream_train.read_record(file_queue) 
            img_batch,label_batch = tf.train.shuffle_batch([img,label],
                                                                           batch_size =FLAGS.train_batch_size,
                                                                           capacity = 2000,
                                                                           min_after_dequeue=1000,
                                                                           num_threads=1
                                                                           ) 
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = self.sess, coord=coord) 
            try:
                for iter_sel in xrange(FLAGS.train_iter):
                        imgs,labels=self.sess.run([img_batch,label_batch])
                        feed_dict = {self.img_batch:imgs,self.img_labels:labels,self.is_train:True}
                        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
                        if iter_sel%FLAGS.display_gap==0:
                            print 'iteraion = %d  loss = %.6f\n'%(iter_sel,loss)
                        if iter_sel%FLAGS.valid_gap==0:
                            val_acc,val_loss = self.validation()
                            print 'do validation on valid dataset:\n validation accuracy = %.4f, validation loss = %.4f'%(val_acc,val_loss)
            except tf.errors.OutOfRangeError:
                print 'input stream was exausted  by wxn \n'
            finally:
                coord.request_stop()          
            coord.request_stop()
            coord.join(threads)
        self.sess.close()
            
            
     
    def build_graph(self):
        self.core = ResNet()
        self.loss = self.core.loss
        self.is_train = self.core.is_train
        self.optimizer = self.core.optimizer
        self.img_batch = self.core.img_batch
        self.img_labels = self.core.img_labels
        self.acc = self.core.acc
         
    def validation(self):
        acc=np.zeros(FLAGS.valid_iter,np.float)
        loss = np.zeros(FLAGS.valid_iter,np.float)
        file_q = tf.train.string_input_producer(["valid_zhiwu_record"])
        image,label = self.stream_val.read_record(file_q) 
        img_batch,label_batch = tf.train.batch([image,label],
                                                      batch_size =FLAGS.valid_batch_size,
                                                      capacity = 20000,
                                                      min_after_dequeue=10000,
                                                      num_threads=1 ) 
        coord_val = tf.train.Coordinator()
        threads_val = tf.train.start_queue_runners(sess = self.sess, coord=coord_val) 
        try:
            for iter_sel in xrange(FLAGS.valid_iter):
                imgs,labels=self.sess.run([img_batch,label_batch])
                feed_dict = {self.img_batch:imgs,self.img_labels:labels,self.is_train:False}
                loss[iter_sel],acc[iter_sel]= self.sess.run([self.loss,self.acc],feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            print 'input stream was exausted  by wxn \n'
        finally:
            coord_val.request_stop()
        coord_val.request_stop()
        coord_val.join(threads_val)
        return np.mean(acc),np.sum(loss)
           
            
                                                                

train_label_list = '/space2/lechao/MyWork/myself/train.txt'
val_label_list = '/space2/lechao/MyWork/myself/valid.txt'
#test_label_list = '../11.lstm-test/label-test.txt'
file_root_train = '/space2/lechao/MyWork/myself/train'
file_root_valid = '/space2/lechao/MyWork/myself/valid'


stream_train = Stream(train_label_list,file_root_train,"train_zhiwu_record")
stream_val = Stream(val_label_list,file_root_valid,"valid_zhiwu_record")
predictor = Predictor(stream_train, stream_val)
predictor.train()