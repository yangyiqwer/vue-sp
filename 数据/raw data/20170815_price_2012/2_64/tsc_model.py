import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import static_rnn

def read_form_csv():
  res = []
  time_step = 30  
  f=open('000001SZ_2012.csv')
  df=pd.read_csv(f)
  price = np.array(df['price'])
  for i in range(len(price) - time_step):
    tmp = []
    if price[i + time_step] - price[i + time_step - 1] > 0:
      tmp.append(1)
    else:
      tmp.append(0)
    for j in range(time_step):
      tmp.append(price[i + j])
    res.append(tmp.copy())
  return np.array(res)

def load_data():
  need_test = True
  """Input:
  direc: location of the UCR archive
  ratio: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  #data_train = np.loadtxt('500cnn_train.txt',delimiter=',')
  #data_test_val = np.loadtxt('500cnn_test.txt',delimiter=',')
  #data = np.loadtxt(os.path.join(os.getcwd(),'000001SZ_all_price.txt'),delimiter=',')
  data = read_form_csv()
  if(need_test):
    data_test_val = data[-100:]
    data_train = data[:-100]
  else:
    data_train = data

  '''
  ratio = (ratio*N).astype(np.int32)
  ind = np.random.permutation(N)
  X_train = DATA[ind[:ratio[0]],1:]
  X_val = DATA[ind[ratio[0]:ratio[1]],1:]
  X_test = DATA[ind[ratio[1]:],1:]
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  y_train = DATA[ind[:ratio[0]],0]-1
  y_val = DATA[ind[ratio[0]:ratio[1]],0]-1
  y_test = DATA[ind[ratio[1]:],0]-1
  return X_train,X_test,y_train,y_test
  '''

  X_train = data_train[:, 1:]
  y_train = data_train[:, 0]
  X_test = data_test_val[:, 1:]
  y_test = data_test_val[:, 0]
  return X_train,X_test,y_train,y_test

def sample_batch(X_train,y_train,batch_size):
  """ Function to sample a batch for training"""
  N,data_len = X_train.shape
  ind_N = np.random.choice(N,batch_size,replace=False)
  X_batch = X_train[ind_N]
  y_batch = y_train[ind_N]
  return X_batch,y_batch


class Model():
  def __init__(self,config):
    
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    max_grad_norm = config['max_grad_norm']
    self.batch_size = config['batch_size']
    sl = config['sl']
    learning_rate = config['learning_rate']
    num_classes = config['num_classes']
    """Place holders"""
    self.input = tf.placeholder(tf.float32, [None, sl], name = 'input')
    #self.input = tf.placeholder(tf.float32, [None, sl, input_size], name = 'input')
    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    self.keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

    with tf.name_scope("LSTM_setup") as scope:
      def single_cell():
        return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size),output_keep_prob=self.keep_prob)

      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
      initial_state = cell.zero_state(self.batch_size, tf.float32)
    
    #!!!modify
    input_list = tf.unstack(tf.expand_dims(self.input,axis=2),axis=1)
    outputs,_ = static_rnn(cell, input_list, dtype=tf.float32)

    output = outputs[-1]


    #Generate a classification from the last cell_output
    #Note, this is where timeseries classification differs from sequence to sequence
    #modelling. We only output to Softmax at last time step
    with tf.name_scope("Softmax") as scope:
      with tf.variable_scope("Softmax_params"):
        softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
        softmax_b = tf.get_variable("softmax_b", [num_classes])
      logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
      #Use sparse Softmax because we have mutually exclusive classes
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.labels,name = 'softmax')
      self.cost = tf.reduce_sum(loss) / self.batch_size
    with tf.name_scope("Evaluating_accuracy") as scope:
      correct_prediction = tf.equal(tf.argmax(logits,1),self.labels)
      self.res = tf.argmax(logits,1)
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      #h1 = tf.summary.scalar('accuracy',self.accuracy)
      #h2 = tf.summary.scalar('cost', self.cost)


    """Optimizer"""
    with tf.name_scope("Optimizer") as scope:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),max_grad_norm)   #We clip the gradients to prevent explosion
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients = zip(grads, tvars)
      self.train_op = optimizer.apply_gradients(gradients)
      # Add histograms for variables, gradients and gradient norms.
      # The for-loop loops over all entries of the gradient and plots
      # a histogram. We cut of
      # for gradient, variable in gradients:  #plot the gradient of each trainable variable
      #       if isinstance(gradient, ops.IndexedSlices):
      #         grad_values = gradient.values
      #       else:
      #         grad_values = gradient
      #
      #       tf.summary.histogram(variable.name, variable)
      #       tf.summary.histogram(variable.name + "/gradients", grad_values)
      #       tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

    #Final code for the TensorBoard
    self.merged = tf.summary.merge_all()
    self.init_op = tf.global_variables_initializer()
    print('Finished computation graph')



