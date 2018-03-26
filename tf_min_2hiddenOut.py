# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:09:45 2018

@author: Sirius

# 对于每天的分钟数据进行预处理
# 对close进行预处理，(y - y_min)/(y_max - y_min) 对结果没有影响
# 用adam优化，预测的不在一条直线上了
# 不加relu的激活层看来是不行的
debug1: init_bias = np.array([np.mean(train_Y)],dtype = 'float32') 如果不设置float32，后面无法计算
sight: 每一部分聚集的点有特征，应该进行分段学习
首先：看那些拐点的权重,bias发生了什么变化
加batch，按顺序学习看是否有改进
1.分段多了
2.收敛速度快
缺点：计算慢
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

TradingDay = '20180316'
symbol = 'IC1803'
dbClient = pymongo.MongoClient('localhost', 27017)
collection = dbClient['VnTrader_1MIN_ALL_1106'][symbol]
livedata = collection.find({'TradingDay':TradingDay}).sort('datetime',pymongo.ASCENDING)     
df = pd.DataFrame(list(livedata))
df = df[['close','volume']]
df.index = range(len(df))
close = np.array(df['close'])
volume = np.array(df['volume'])
index_array = np.arange(len(close))
# train_X = np.transpose(np.array([index_array,volume]))
# train_Y = np.array([[t] for t in close])

train_X = np.array([[t] for t in index_array])# [:80]
train_Y = np.array([[t] for t in close])# [:80]
init_bias = np.array([np.mean(train_Y)],dtype = 'float32')

'''
ymin,  ymax = train_Y.min(), train_Y.max()
train_Y = (train_Y - ymin)/(ymax - ymin)
'''
'''
ind = [i for i in range(len(train_X))]
shuffle(ind)
train_X = train_X[ind]
train_Y = train_Y[ind]
'''

##############################################################################
# Parameters
learning_rate = 0.001
training_epochs = 50000
batch_size = 25

# tf Graph Input
x = tf.placeholder(tf.float32, shape=(None,1), name = 'x')
y = tf.placeholder(tf.float32, shape=(None,1), name = 'y')

hidden_num = 10
# Set model weights
weights1 = tf.Variable(tf.random_normal([1,hidden_num],stddev = 1,seed = 1),name = 'weights1')
biases1 = tf.Variable(tf.constant(0.0,shape=[1,hidden_num]),name = 'biases1')

weights2 = tf.Variable(tf.random_normal([hidden_num,1],stddev = 1,seed = 2),name = 'weights2')
biases2 = tf.Variable(tf.constant(init_bias,shape=[1,1]),name = 'biases2')

regularizers = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(biases2)


# hidden_out = tf.add(tf.matmul(x, weights1),biases1)
hidden_out = tf.nn.relu(tf.add(tf.matmul(x, weights1),biases1))
# hidden_out = tf.sigmoid(tf.add(tf.matmul(x, weights1),biases1))
y_ = tf.add(tf.matmul(hidden_out,weights2),biases2)

cost = tf.losses.mean_squared_error(y,y_) # + 1e-4 * regularizers
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init_op = tf.global_variables_initializer()
accuracy = tf.reduce_mean(tf.square(tf.subtract(y,y_)))


# run
with tf.Session() as sess:
  sess.run(init_op)
  COST = []
  total_batch = int(len(train_Y) / batch_size)
  for epoch in range(training_epochs):
      for i in range(total_batch):
          batch_x, batch_y = train_X[i * batch_size:min(i * batch_size + batch_size, len(train_X))], \
                             train_Y[i * batch_size:min(i * batch_size + batch_size, len(train_Y))]
          _, c = sess.run([optimizer,cost], feed_dict={x: batch_x, y: batch_y}) # _ : store previous result
      tmp = sess.run(cost, feed_dict={x: train_X, y: train_Y})
      COST.append(tmp)
      if epoch%1000==0:
         print('epoch',epoch)
         print('cost',tmp)
         y_pred = sess.run(y_,feed_dict={x: train_X})
         plt.scatter(train_X,train_Y)
         plt.scatter(train_X,y_pred)
         plt.show()
  y_pred = sess.run(y_,feed_dict={x: train_X})
  



# plt.plot(np.arange(training_epochs), COST)