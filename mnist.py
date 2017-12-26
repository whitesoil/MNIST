
# coding: utf-8

# In[ ]:


import gzip
import pickle
import os
import numpy as np

key_file = {}

save_file = "./mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

        
def _load_label(file_name):
    file_path ="./" + file_name

    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    return labels

def _load_img(file_name):
    file_path = "./" + file_name
 
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['testall_img'] = _load_img(key_file['testall_img'])
    return dataset

def init_mnist():
    dataset = _convert_numpy()
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(train_img,train_lab,final_img,normalize=True, flatten=True, one_hot_label=False):
    key_file['train_img'] = train_img
    key_file['train_label'] = train_lab
    key_file['testall_img'] = final_img
    
    init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img','testall_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
    
    if not flatten:
         for key in ('train_img','testall_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
            
    return (dataset['train_img'], dataset['train_label'], dataset['testall_img'])

if __name__ == '__main__':
    init_mnist()

