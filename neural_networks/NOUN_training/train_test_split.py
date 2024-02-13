import pickle
from pdb import set_trace as bp
import numpy as np
import random

###########################################################
with open('Data/noun_xception_feats.pkl', 'rb') as fp:
    image_data = pickle.load(fp)
    
key1 = list(image_data.keys())

with open('Data/audio_features_noun.pkl', 'rb') as fp:
    audio_data = pickle.load(fp)

###############################################
classes = open('pairs.txt','r').read().split('\n')
classes = classes[:-1]

#random.shuffle(classes)   #uncomment to randomise the pairings also

base_path = 'Data/NOUN_train_test_split/'
###############################################


image_train = []
image_test = []
audio_train = []
audio_test = []
image_val =[]
audio_val = []

n_sessions = 6
n_class_each_session = 10

n_total_img =10
n_total_aud =12

n_train = 1
n_val = 2

###############################################################


for session in range(n_sessions):
    
    image_train_temp = {}
    image_test_temp = {}
    
    audio_train_temp = {}
    audio_test_temp = {}
    
    image_val_temp = {}
    audio_val_temp = {}
    
    for i in range(n_class_each_session):
        
        n_list = [k for k in range(n_total_img)]
        n_list = np.array(n_list)

        random.shuffle(n_list)
        
        index = session*n_class_each_session + i
    
        temp_data_img = image_data[key1[index]]
    
        train_index = n_list[0:n_train]
        val_index= n_list[n_train:n_train+n_val]
        test_index = n_list[n_train+n_val:]
    
        train_temp = temp_data_img[train_index,:]
    
        train_temp = np.reshape(train_temp, [n_train, 2048])   #only if there is one training sample
        val_temp = temp_data_img[val_index,:]
    
        test_temp = temp_data_img[test_index,:]
    
        image_train_temp[classes[index]] = train_temp
        image_val_temp[classes[index]] = val_temp
        image_test_temp[classes[index]] = test_temp
        
        #print(key1[index], classes[index])
        
        
    for i in range(n_class_each_session):
        
        n_list = [k for k in range(n_total_aud)]
        n_list = np.array(n_list)

        random.shuffle(n_list)
        
        index = session*n_class_each_session + i
    
        temp_data_aud = audio_data[classes[index]]
    
        train_index = n_list[0:n_train]
        val_index= n_list[n_train:n_train+n_val]
        test_index = n_list[n_train+n_val:]
    
        train_temp = temp_data_aud[train_index,:,:]
    
        train_temp = np.reshape(train_temp, [n_train, 229,80])   #only if there is one training sample
        val_temp = temp_data_aud[val_index,:]
    
        test_temp = temp_data_aud[test_index,:,:]
    
        audio_train_temp[classes[index]] = train_temp
        audio_val_temp[classes[index]] = val_temp
        audio_test_temp[classes[index]] = test_temp    
        
    image_train.append(image_train_temp)
    image_test.append(image_test_temp)
    
    audio_train.append(audio_train_temp)
    audio_test.append(audio_test_temp)
    
    image_val.append(image_val_temp)
    audio_val.append(audio_val_temp)
    

###########################3

with open('Data/NOUN_train_test_split/image_features_train.pkl'.format(base_path), 'wb') as file:
    pickle.dump(image_train, file)
    
with open('Data/NOUN_train_test_split/audio_features_train.pkl'.format(base_path), 'wb') as file:
    pickle.dump(audio_train, file)

with open('Data/NOUN_train_test_split/image_features_test.pkl'.format(base_path), 'wb') as file:
    pickle.dump(image_test, file)
    
with open('Data/NOUN_train_test_split/audio_features_test.pkl'.format(base_path), 'wb') as file:
    pickle.dump(audio_test, file)
    
with open('Data/NOUN_train_test_split/image_features_val.pkl'.format(base_path), 'wb') as file:
    pickle.dump(image_val, file)
    
with open('NOUN_train_test_split/audio_features_val.pkl'.format(base_path), 'wb') as file:
    pickle.dump(audio_val, file)