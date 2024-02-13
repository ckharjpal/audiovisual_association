import matplotlib
matplotlib.use('Agg')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import JointNet
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import pandas as pd

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

##########################################################################

BATCH_SIZE = 128

NUM_CLASSES = 576

vec_length = 512  # length of d_dim representation

#############################################

path = '/home/chandrakanth/Audio-Visual-Deep-Multimodal-Networks-master/Data/'

with open(path+'audio_features_train.pkl','rb') as f:
	audio_train = pickle.load(f)

with open(path+'audio_features_val.pkl','rb') as f:
	audio_val = pickle.load(f)

with open(path+'image_features_train.pkl','rb') as f:
	img_train = pickle.load(f)

with open(path+'image_features_val.pkl','rb') as f:
	img_val = pickle.load(f)

###############################################################

MAX_LEN = audio_train[list(audio_train)[0]].shape[1]

classes = open(path+'../classes.txt','r').read().split('\n')
classes = classes[:-1]
label_ind = {c:i for i,c in enumerate(classes)}

#############################################

def generator(df,grd):
	num_samples = len(df)

	lab2idx = label_ind

	i = 0
	while True:

		if i%num_samples>=0:
			np.random.shuffle(df)             
			i = 0

		anchor_labels = df[i:i+BATCH_SIZE,1]

		if grd==0:
			groundings = np.zeros(BATCH_SIZE)
			anchor_img = df[i:i+BATCH_SIZE,0]
			anchor_img = np.array([i for i in anchor_img])
			anchor_aud = np.zeros((BATCH_SIZE,MAX_LEN,80))

		else:
			groundings = np.ones(BATCH_SIZE)
			anchor_aud = df[i:i+BATCH_SIZE,0]
			anchor_aud = np.array([i for i in anchor_aud])
			anchor_img = np.zeros((BATCH_SIZE,2048))

		class_mask = np.zeros((BATCH_SIZE, NUM_CLASSES))
		class_mask_bar = np.ones((BATCH_SIZE, NUM_CLASSES))
  

		for s,x in enumerate(groundings):                
			label = anchor_labels[s]
			class_mask[s][lab2idx[label]] = 1
			class_mask_bar[s][lab2idx[label]] = 0
		
		yield [groundings, 1-groundings, np.array(anchor_aud), np.array(anchor_img), np.array(class_mask), np.array(class_mask_bar)], np.zeros(BATCH_SIZE)
  
		i += BATCH_SIZE
  

def createdf(data):
	df = []
	for i in data:
		for j in data[i]:
			df.append([j,i])
	df = np.array(df)
	return df


df_img_train = createdf(img_train)
df_img_val = createdf(img_val)
df_aud_train = createdf(audio_train)
df_aud_val = createdf(audio_val)


print(df_img_train.shape)
print(df_img_val.shape)
print(df_aud_train.shape)
print(df_aud_val.shape)


#################################################################
print('Training IMAGES: ')
#Phase1: Train only on images

folder = "Saved_models/proxy_models_new"

filepath = folder+"/saved-proxy-imagenetwork.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True, period=1)
callbacks_list = [checkpoint]

jointnet = JointNet(np.zeros((vec_length, NUM_CLASSES)), True)
model = jointnet.model
print(model.summary())

history = model.fit_generator(generator(df_img_train, 0), epochs=40,steps_per_epoch = len(df_img_train)/BATCH_SIZE, callbacks=callbacks_list, initial_epoch=0,validation_data = generator(df_img_val,0),validation_steps = len(df_img_val)/BATCH_SIZE)

################################################################
print('Training AUDIO: ')

filepath = folder+"/saved-proxy-audionetwork.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True, period=1)
callbacks_list = [checkpoint]

# # Phase2: Train only on audio
jointnet = JointNet(np.zeros((vec_length, NUM_CLASSES)), False)
model = jointnet.model
model.load_weights('Saved_models/proxy_models_new/saved-proxy-imagenetwork.hdf5', by_name=True)    # Choose epoch weights to load

# model.layers[5].layers[1].trainable=False
# model.layers[-4].trainable=False
# model.compile(loss=jointnet.identity_loss, optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-5))

print('Trainable weights: ',len(model.trainable_weights),[i.shape for i in model.trainable_weights])

history = model.fit_generator(generator(df_aud_train, 1), epochs=300, steps_per_epoch = len(df_aud_train)/BATCH_SIZE, callbacks=callbacks_list, initial_epoch=0,validation_data = generator(df_aud_val,1),validation_steps = len(df_aud_val)/BATCH_SIZE)


############################################