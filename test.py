# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:01:59 2017

@author: lechao
"""

from module import *
import tensorflow  as tf
if __name__ == "__main__":
    a =tf.zeros([4,4,4,4])
    c2 = Conv2d([3,3,4,12])  
    b = c2.apply(a)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(b)