import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
'''
# Hyperparameters
num_filt_1 = 15  # Number of filters in first conv layer
num_filt_2 = 8  # Number of filters in second conv layer
num_filt_3 = 8  # Number of filters in thirs conv layer
num_fc_1 = 40  # Number of neurons in hully connected layer
max_iterations = 20000
batch_size = 100
dropout = 0.5  # Dropout rate in the fully connected layer
plot_row = 5  # How many rows do you want to plot in the visualization
regularization = 1e-4
learning_rate = 2e-3
input_norm = False  # Do you want z-score input normalization?
'''
"""Hyperparameters"""
num_filt_1 = 16     #Number of filters in first conv layer
num_filt_2 = 14      #Number of filters in second conv layer
num_filt_3 = 8      #Number of filters in thirs conv layer
num_fc_1 = 40       #Number of neurons in hully connected layer
max_iterations = 20000
batch_size = 100
dropout = 0.5       #Dropout rate in the fully connected layer
plot_row = 5        #How many rows do you want to plot in the visualization
learning_rate = 2e-5
input_norm = False     # Do you want z-score input normalization?
regularization = 1e-4
# load data

def read_form_csv():
  res = []
  time_step = 30  
  f=open('000001SZ_2014.csv')
  df=pd.read_csv(f)
  price = np.array(df['price'])
  for i in range(1, len(price) - time_step):
    tmp = []
    if price[i + time_step] - price[i + time_step - 1] > 0:
      tmp.append(1)
    else:
      tmp.append(0)
    for j in range(time_step):
      if price[i + j] - price[i + j - 1] > 0:
        tmp.append(1)
      else:
        tmp.append(0)
    res.append(tmp.copy())
  return np.array(res) 

data = read_form_csv()
data_test_val = data[-100:]
data_train = data[:-100]

'''

time_step = 30

f = open('500_train.csv')
df = pd.read_csv(f)
date = np.array(df['date'])
price = np.array(df['price'])
y_train_list = []
x_train_list = []
for i in range(len(date) - time_step):
    if price[i + time_step] - price[i + time_step -1] > 0:
        y_train_list.append(1)
    else:
        y_train_list.append(2)
    x_train_list.append(price[i:i+time_step])
X_train = np.array(x_train_list)
y_train = np.array(y_train_list)
y_train = y_train.reshape(len(y_train_list),)
#print(X_train)
print(X_train.shape)
print(y_train.shape)
print(y_train)
N = X_train.shape[0]
D = X_train.shape[1]

f = open('500_test.csv')
df = pd.read_csv(f)
date = np.array(df['date'])
price = np.array(df['price'])
y_test_list = []
x_test_list = []
for i in range(len(date) - time_step):
    if price[i + time_step] - price[i + time_step -1] > 0:
        y_test_list.append(1)
    else:
        y_test_list.append(2)
    x_test_list.append(price[i:i+time_step])
X_test = np.array(x_test_list)
y_test = np.array(y_test_list)
y_test = y_test.reshape(len(y_test_list),)

'''

# split training and testing data
X_train = data_train[:,1:]
X_test = data_test_val[:,1:]
N = X_train.shape[0]
Ntest = X_test.shape[0]
D = X_train.shape[1]
y_train = data_train[:,0]
y_test = data_test_val[:,0]


# normalize x and y
num_classes = len(np.unique(y_train))
'''
base = np.min(y_train)  #Check if data is 0-based
if base != 0:
    y_train -= base
    y_test -= base
'''

if input_norm:
    mean = np.mean(X_train,axis=0)
    variance = np.var(X_train,axis=0)
    X_train -= mean
    #The 1e-9 avoids dividing by zero
    X_train /= np.sqrt(variance)+1e-9
    X_test -= mean
    X_test /= np.sqrt(variance)+1e-9

#epochs = np.floor(batch_size*max_iterations / N)
#print('Train with approximately %d epochs' %(epochs))

# place for the input variables
x = tf.placeholder("float", shape=[None, D], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
keep_prob = tf.placeholder("float")
bn_train = tf.placeholder(tf.bool)  #Boolean value to guide batchnorm

class ConvolutionalBatchNormalizer(object):
  """Helper class that groups the normalization logic and variables.        

  Use:                                                                      
      ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
      bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)           
      update_assignments = bn.get_assigner()                                
      x = bn.normalize(y, train=training?)                                  
      (the output x will be batch-normalized).                              
  """

  def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm):
    self.mean = tf.Variable(tf.constant(0.0, shape=[depth]),
                            trainable=False)
    self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
                                trainable=False)
    self.beta = tf.Variable(tf.constant(0.0, shape=[depth]), name = 'beta')
    self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]),name = 'gamma')
    self.ewma_trainer = ewma_trainer
    self.epsilon = epsilon
    self.scale_after_norm = scale_after_norm

  def get_assigner(self):
    """Returns an EWMA apply op that must be invoked after optimization."""
    return self.ewma_trainer.apply([self.mean, self.variance])

  def normalize(self, x, train=True):
    """Returns a batch-normalized version of x."""
    if train is not None:
      mean, variance = tf.nn.moments(x, [0,1,2])
      assign_mean = self.mean.assign(mean)
      assign_variance = self.variance.assign(variance)
      with tf.control_dependencies([assign_mean, assign_variance]):
        return tf.nn.batch_norm_with_global_normalization(
            x, mean, variance, self.beta, self.gamma,
            self.epsilon, self.scale_after_norm)
    else:
      mean = self.ewma_trainer.average(self.mean)
      variance = self.ewma_trainer.average(self.variance)
      local_beta = tf.identity(self.beta)
      local_gamma = tf.identity(self.gamma)
      return tf.nn.batch_norm_with_global_normalization(
          x, mean, variance, local_beta, local_gamma,
          self.epsilon, self.scale_after_norm)

# w and b and conv function
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("Reshaping_data") as scope:
    x_image = tf.reshape(x, [-1,D,1,1])


# Build the graph
# ewma is the decay for which we update the moving average of the 
# mean and variance in the batch-norm layers
with tf.name_scope("Conv1") as scope:
    W_conv1 = weight_variable([4, 1, 1, num_filt_1], 'Conv_Layer_1')
    b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
    a_conv1 = conv2d(x_image, W_conv1) + b_conv1
  
with tf.name_scope('Batch_norm_conv1') as scope:
    ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
    bn_conv1 = ConvolutionalBatchNormalizer(num_filt_1, 0.001, ewma, True)           
    update_assignments = bn_conv1.get_assigner() 
    a_conv1 = bn_conv1.normalize(a_conv1, train=bn_train) 
    h_conv1 = tf.nn.relu(a_conv1)
  
with tf.name_scope("Conv2") as scope:
    W_conv2 = weight_variable([4, 1, num_filt_1, num_filt_2], 'Conv_Layer_2')
    b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
    a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
  
with tf.name_scope('Batch_norm_conv2') as scope:
    bn_conv2 = ConvolutionalBatchNormalizer(num_filt_2, 0.001, ewma, True)           
    update_assignments = bn_conv2.get_assigner() 
    a_conv2 = bn_conv2.normalize(a_conv2, train=bn_train) 
    h_conv2 = tf.nn.relu(a_conv2)
    
with tf.name_scope("Conv3") as scope:
    W_conv3 = weight_variable([4, 1, num_filt_2, num_filt_3], 'Conv_Layer_3')
    b_conv3 = bias_variable([num_filt_3], 'bias_for_Conv_Layer_3')
    a_conv3 = conv2d(h_conv2, W_conv3) + b_conv3
  
with tf.name_scope('Batch_norm_conv3') as scope:
    bn_conv3 = ConvolutionalBatchNormalizer(num_filt_3, 0.001, ewma, True)           
    update_assignments = bn_conv3.get_assigner() 
    a_conv3 = bn_conv3.normalize(a_conv3, train=bn_train) 
    h_conv3 = tf.nn.relu(a_conv3)

with tf.name_scope("Fully_Connected1") as scope:
    W_fc1 = weight_variable([D*num_filt_3, num_fc_1], 'Fully_Connected_layer_1')
    b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
    h_conv3_flat = tf.reshape(h_conv3, [-1, D*num_filt_3])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
  
with tf.name_scope("Fully_Connected2") as scope:
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = tf.Variable(tf.truncated_normal([num_fc_1, num_classes], stddev=0.1),name = 'W_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]),name = 'b_fc2')
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  
 
with tf.name_scope("SoftMax") as scope:
    regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
                  tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) + 
                  tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + 
                  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h_fc2,labels=y_)
    cost = tf.reduce_sum(loss) / batch_size
    cost += regularization*regularizers


# define train optimizer
with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    #We clip the gradients to prevent explosion
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)

    numel = tf.constant([[0]])
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient
      
      numel +=tf.reduce_sum(tf.size(variable))  
with tf.name_scope("Evaluating_accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(h_fc2,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# run session and evaluate performance
time = 0
while(time < 1):
    perf_collect = np.zeros((3,int(np.floor(max_iterations /100))))
    #merged = tf.merge_all_summaries()
    with tf.Session() as sess:
        #writer = tf.train.SummaryWriter("/home/carine/Desktop/eventlog/", sess.graph_def)
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(tf.global_variables())
        step = 0  # Step is a counter for filling the numpy array perf_collect
        for i in range(max_iterations):#training process
            batch_ind = np.random.choice(N,batch_size,replace=False)
            
            if i==0:
                acc_test_before = sess.run(accuracy, feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
            if i%100 == 0:
                #Check training performance
                result,train_cost = sess.run([accuracy,cost],feed_dict = { x: X_train, y_: y_train, keep_prob: 1.0, bn_train : False})
                perf_collect[1,step] = result 
                acc_train = result
                #Check validation performance
                predict,result,test_cost = sess.run([tf.argmax(h_fc2,1),accuracy,cost],feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
                acc = result
                perf_collect[0,step] = acc
                print(predict)    
                print(" Training accuracy at %s out of %s is %5.3f(%5.3f),cost is %5.3f(%5.3f)" % (i,max_iterations, acc_train, acc, train_cost, test_cost))
                step +=1
            sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train : True})
            
              #training process done!
        result = sess.run([accuracy,numel], feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
      
        predict=sess.run(tf.argmax(h_fc2,1), feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
        #pred_summ=tf.scalar_summary("prediction", predict)
        
        print("pred "+"real")
        right = 0
        for x in range(0,len(predict)):
            print(str(predict[x]+1)+"    "+str(int(y_test[x]+1)))
            if predict[x] == y_test[x]:
                right += 1
        acc_test = result[0]
        
        print('time:' + str(time) + ' ', end="")
        print('The accuracy on the test data is %.3f, before training was %.3f' %(acc_test,acc_test_before))
        print(right / 100)
        #print('The network has %s trainable parameters'%(result[1]))
        if(right / 100 > 1):
            model_datapath = os.path.join(os.getcwd(), 'cnn_model', str(time) + ' ' + str(right))
            if not os.path.exists(model_datapath):
                os.mkdir(model_datapath)
            model_name = 'ckp'
            saver.save(sess,os.path.join(model_datapath, model_name))
    #writer.flush()
    time += 1

# show the graph of validation accuracy
# 2 for drop or same, 1 for rise 

'''
plt.figure()
plt.plot(perf_collect[0],label='Valid accuracy')
plt.plot(perf_collect[1],label = 'Train accuracy')
plt.axis([0, step, 0, np.max(perf_collect)])
plt.show()
plt.figure()
'''
