# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:15:43 2018

@author: Sirius
"""

# An example from tensorflow website
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from random import randint
from random import uniform
from random import shuffle
from random import seed

import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

TradingDay = '20180316'
symbol = 'IC1803'
dbClient = pymongo.MongoClient('localhost', 27017)
collection = dbClient['VnTrader_1MIN_ALL_1106'][symbol]
# livedata = collection.find({'TradingDay':TradingDay}).sort('datetime',pymongo.ASCENDING)
flt = {'datetime':{'$gte':datetime(2018,1,1,21,00)}}
livedata = collection.find(flt).sort('datetime',pymongo.ASCENDING)    
df = pd.DataFrame(list(livedata))
df = df[['close','volume']]
df.index = range(len(df))
close = np.array(df['close'])
volume = np.array(df['volume'])
index_array = np.arange(len(close))


train_X = np.array([[t] for t in index_array])# [:100]
train_Y = np.array([[t] for t in close])# [:100]

seed(2)
ind = [i for i in range(len(train_X))]
shuffle(ind)
train_X = train_X[ind]
train_Y = train_Y[ind]

# train_Y = train_Y[::-1]
init_bias = np.array([np.mean(train_Y)],dtype = 'float32')
# plt.title('after shuffle')

##############################################################################
# Parameters
learning_rate = 0.01
training_epochs = 10000

# tf Graph Input
x = tf.placeholder(tf.float32, shape=(None,1), name = 'x')
y = tf.placeholder(tf.float32, shape=(None,1), name = 'y')

hidden_num = 200
# Set model weights
weights1 = tf.Variable(tf.random_normal([1,hidden_num],stddev = 0.1,seed = 1),name = 'weights1')
biases1 = tf.Variable(tf.constant(0.0,shape=[1,hidden_num]),name = 'biases1')

weights2 = tf.Variable(tf.random_normal([hidden_num,hidden_num],stddev = 0.1,seed = 2),name = 'weights2')
biases2 = tf.Variable(tf.constant(0.0,shape=[1,hidden_num]),name = 'biases2')

weights3 = tf.Variable(tf.random_normal([hidden_num,hidden_num],stddev = 0.1,seed = 3),name = 'weights3')
biases3 = tf.Variable(tf.constant(0.0,shape=[1,hidden_num]),name = 'biases3')

weights4 = tf.Variable(tf.random_normal([hidden_num,1],stddev = 0.1,seed = 4),name = 'weights4')
biases4 = tf.Variable(tf.constant(init_bias,shape=[1,1]),name = 'biases4')


hidden_out1 = tf.nn.relu(tf.add(tf.matmul(x, weights1),biases1))
hidden_out2 = tf.nn.relu(tf.add(tf.matmul(hidden_out1, weights2),biases2))
hidden_out3 = tf.nn.relu(tf.add(tf.matmul(hidden_out2, weights3),biases3))

y_ = tf.add(tf.matmul(hidden_out3,weights4),biases4)


cost = tf.losses.mean_squared_error(y,y_) # + 1e-4 * regularizers
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init_op = tf.global_variables_initializer()
accuracy = tf.reduce_mean(tf.square(tf.subtract(y,y_)))


# run
with tf.Session() as sess:
  sess.run(init_op)
  COST = []
  for epoch in range(training_epochs):
      sess.run(optimizer, feed_dict={x: train_X, y: train_Y})
      tmp = sess.run(cost, feed_dict={x: train_X, y: train_Y})
      COST.append(tmp)
      if epoch%1000==0:
         # tmp = sess.run(cost, feed_dict={x: train_X, y: train_Y})
         # COST.append(tmp)
         print('epoch',epoch)
         print('cost',tmp)
         y_pred = sess.run(y_,feed_dict={x: train_X})
         fig = plt.figure(figsize = (12,8))
         ax = fig.add_subplot(111)  
         ax.scatter(train_X,train_Y)
         ax.scatter(train_X,y_pred)
         plt.show()


