# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:38:25 2017

@author: lechao
"""
import logging 
import os 
import tensorflow as tf
from PIL import Image
import numpy as np
logging.basicConfig(level=logging.DEBUG, 
                    format = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt = '%a, %d %b %Y %H:%M:%S',
                    filename = 'stream.log',
                    filemode = 'w')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)  
# generate tf.records file (big dataset) or generate stream in RAM (small dataset)  

class Config_Record():
    def __init__(self,is_shuffle=False,resize_width=0,resize_height=0):
        self.is_shuffle = is_shuffle
        self.resize_width = resize_width
        self.resize_height = resize_height
        
class Stream():
    def __init__(self,train_label_list,file_root, record_name, generate_record=False,config_record = Config_Record(True,128,128)):
        self.train_label_list = train_label_list
        self.file_root =  file_root if file_root[-1]=='/' else file_root+'/'
        self.gen_rec = generate_record
        self.record_name = record_name
        self.config_record = config_record
        self.num = 0
        
    def create_record(self):
        writer = tf.python_io.TFRecordWriter(self.record_name)
        
        print "start create %s \n"% self.record_name
        if os.path.exists(self.train_label_list)==False:
            logging.error(self.train_label_list+'don\'t exist\n' )
        with open(self.train_label_list) as f:
            lines = f.readlines()
        print "total %d files"% len(lines)
        order = range(len(lines))
        if self.config_record.is_shuffle == True:
            np.random.shuffle(order)
        for i in order:
            if self.num%1000 == 0:
                print "process %d files\n"% self.num
            line_split = lines[i].split()
            if len(line_split)<2 or os.path.exists(self.file_root+line_split[0])==False:
                continue
            self.num+=1
            img = Image.open(self.file_root+line_split[0])
            img = img.resize((self.config_record.resize_height,self.config_record.resize_width))
            img_raw = img.tobytes()
            label = [int(x) for x in line_split[1:]]
            example = tf.train.Example(features=tf.train.Features(
            feature = {"label":tf.train.Feature(int64_list = tf.train.Int64List(value=label)),
                       "img":tf.train.Feature(bytes_list =tf.train.BytesList(value =[img_raw]))
                       }
                )
            )
            writer.write(example.SerializeToString())
        writer.close()
        info_str = '%d files in %s\n'%(self.num,self.record_name)
        logging.info(info_str)
    def read_record(self, file_queue): 
        if file_queue==None:
            raise ValueError('there isn\'t any record could be loaded!\n')
        reader = tf.TFRecordReader()
        _,serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(serialized_example,
                                               features = {
                                               "label":tf.FixedLenFeature([],tf.int64),
                                               "img":tf.FixedLenFeature([],tf.string),                                           
                                               })
        img = tf.decode_raw(features["img"], tf.uint8)
        img = tf.reshape(img, shape=[self.config_record.resize_height,self.config_record.resize_width,3])
        img = tf.cast(img, tf.float32)*(1. /255) - 0.5
        label = tf.expand_dims(tf.cast(features["label"],tf.int32),-1)
        return img,label

train_label_list = '/space2/lechao/MyWork/myself/train.txt'
val_label_list = '/space2/lechao/MyWork/myself/valid.txt'
#test_label_list = '../11.lstm-test/label-test.txt'
file_root_train = '/space2/lechao/MyWork/myself/train'
file_root_valid = '/space2/lechao/MyWork/myself/valid'
#file_root2 = '../CaffeLSTM-OCR'
#stream_train = Stream(train_label_list,file_root_train,"train_zhiwu_record",True,config)
#stream_test = Stream(test_label_list,file_root2,"test_record",True,config)
#stream_val = Stream(val_label_list,file_root_valid,"valid_zhiwu_record",True,config)
#stream.create_record()
#stream_train.create_record()
#stream_val.create_record()
#stream_test.create_record()
#logging.info('hello')
#logging.debug('thi is a bug')