import numpy as np
import tensorflow as tf  #TF 1.1.0rc1
tf.logging.set_verbosity(tf.logging.ERROR)
#import matplotlib.pyplot as plt
from tsc_model import Model,sample_batch,load_data
import os
#Set these directories
#direc = '/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015'
#summaries_dir = '/home/rob/Dropbox/ml_projects/LSTM_TSC/log_tb'

"""Load the data"""
#ratio = np.array([0.8,0.9]) #Ratios where to split the training and validation set
X_train,X_test,y_train,y_test = load_data()
N,sl = X_train.shape
num_classes = len(np.unique(y_train))

"""Hyperparamaters"""
batch_size = 100
max_iterations = 30000
dropout = 0.5
config = {    'num_layers' :    2,               #number of layers of stacked RNN's
              'hidden_size' :   64,             #memory cells in a layer
              'max_grad_norm' : 5,             #maximum gradient norm during training
              'batch_size' :    batch_size,
              'learning_rate' : .005,
              'sl':             sl,
              'num_classes':    num_classes}



epochs = np.floor(batch_size*max_iterations / N)
print('Train %.0f samples in approximately %d epochs' %(N,epochs))

#Instantiate a model
model = Model(config)

"""Session time"""
sess = tf.Session() #Depending on your use, do not forget to close the session
#writer = tf.summary.FileWriter(summaries_dir, sess.graph)  #writer for Tensorboard
time = 0
model_datapath = os.path.join(os.getcwd(),"lstm_model")
model_name = 'ckp'
dic = {}
for i in range(101):
  dic[i] = 0
best = 0
while time < 1:
  sess.run(model.init_op)

  cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
  acc_train_ma = 0.0

  for i in range(max_iterations):
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size)

    #Next line does the actual training
    cost_train, acc_train,_ = sess.run([model.cost,model.accuracy, model.train_op],feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout})
    cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
    acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
    if i%100 == 1:
    #Evaluate validation performance
      #X_batch, y_batch = X_test, y_test
      #cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
      X_batch, y_batch = sample_batch(X_test,y_test,batch_size)
      res, cost_val,acc_val = sess.run([model.res, model.cost,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
      print(res)
      print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
      #Write information to TensorBoard
      #writer.add_summary(summ, i)
      #writer.flush()
    res, cost_val,acc_val = sess.run([model.res,model.cost,model.accuracy],feed_dict = {model.input: X_test, model.labels: y_test, model.keep_prob:1.0})
  with open("result.txt", "a") as f:  
    f.write("time:")
    f.write(str(time)+" ")
    #epoch = float(i)*batch_size/N
    #f.write(str('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f'%(epoch,acc_val,cost_val)))
    f.write(str(acc_train) + " " + str(acc_val))
    f.write("\n")
    f.write(str(res))
    f.write("\n")
    dic[int(acc_val * 100)] += 1
  if acc_val > best:
    best = acc_val
    saver=tf.train.Saver(tf.global_variables())
    if not os.path.exists(model_datapath):
      os.mkdir(model_datapath)
    saver.save(sess,os.path.join(model_datapath, model_name))
  time += 1
with open("result.txt", "a") as f:  
  for i in range(101):
    if(dic[i] != 0):
      f.write(str(i / 100) + ":" + str(dic[i]) + "\n")
#now run in your terminal:
# $ tensorboard --logdir = <summaries_dir>
# Replace <summaries_dir> with your own dir



