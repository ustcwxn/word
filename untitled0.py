# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def read_record(file_queue):  
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example,
                                           features = {
                                           "label":tf.FixedLenFeature([],tf.int64),
                                           "img":tf.FixedLenFeature([],tf.string),                                           
                                           })
    img = tf.decode_raw(features["img"], tf.uint8)
    img = tf.reshape(img, shape=[128,128,3])
    img = tf.cast(img, tf.float32)*(1. /255) - 0.5
    label = tf.cast(features["label"],tf.int32)
    return img,label
        
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess: 
    with tf.device('/gpu:1'):
        sess.run(tf.initialize_all_variables())
        file_queue = tf.train.string_input_producer(["train_zhiwu_record"])
        img,label = read_record(file_queue)
        img_batch,label_batch = tf.train.shuffle_batch([img,label],
                                                                       batch_size =2,
                                                                       capacity = 200,
                                                                       min_after_dequeue=100,
                                                                       num_threads=3
                                                                       )      
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess =sess,coord=coord)
        img_batch,label_batch = sess.run([img_batch,label_batch])
        print img_batch
        print label_batch
        coord.request_stop()
        coord.join(threads)
        