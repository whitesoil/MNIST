
# coding: utf-8

# In[4]:


import sys, os
import numpy as np
from mnist import load_mnist
from two_layer import TwoLayer

train_img = sys.argv[1]
train_lab = sys.argv[2]
final_img= sys.argv[3]

(i_train, l_train, f_test) = load_mnist(train_img,train_lab,final_img,normalize=True, one_hot_label=True)

network = TwoLayer(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = i_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    i_batch = i_train[batch_mask]
    l_batch = l_train[batch_mask]

    grad = network.gradient(i_batch, l_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(i_batch, l_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(i_train, l_train)
        train_acc_list.append(train_acc)
        print(train_acc)
        
fo = open('prediction.txt','w')

for i in range(0,60000,batch_size):
    f_batch = f_test[i:i+batch_size]
    pred = network.test_accuracy(f_batch)
    for x in range(100):
        fo.write(str(pred[x])+'\n')
        
fo.close()

