import matplotlib
matplotlib.use('Agg')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model_noun import JointNet
import os
import numpy as np
import os
import pickle
from keras.callbacks import ModelCheckpoint
import sys

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
#####################################

vec_length = 576

#session = 0   #goes from 0 to 5
session = int(sys.argv[1])

img_epochs = 20
aud_epochs = 40

#############################################

path = 'Data/NOUN_train_test_split/'

with open(path+'audio_features_train.pkl','rb') as f:
	audio_train_session = pickle.load(f)

with open(path+'image_features_train.pkl','rb') as f:
	img_train_session = pickle.load(f)

with open(path+'image_features_val.pkl','rb') as f:
	img_val_session = pickle.load(f)

with open(path+'audio_features_val.pkl','rb') as f:
	audio_val_session = pickle.load(f)
 

audio_train = audio_train_session[session]
img_train = img_train_session[session]

audio_val = audio_val_session[session]
img_val = img_val_session[session]


MAX_LEN = audio_train[list(audio_train)[0]].shape[1]
###############################################################


classes = list(img_train.keys())

label_ind = {c:i for i,c in enumerate(classes)}

NUM_CLASSES = len(classes)
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
	df = np.array(df, dtype=object)
	return df


df_img_train = createdf(img_train)
df_img_val = createdf(img_val)
df_aud_train = createdf(audio_train)
df_aud_val = createdf(audio_val)


BATCH_SIZE = 5
#################################################################
if session ==0:
    load_filename = 'Saved_models/saved-proxy-audionetwork.hdf5'
    if os.path.exists('Saved_models/noun_model/saved-proxy-audionetwork_0.hdf5'):
        load_filename = 'Saved_models/noun_model/saved-proxy-audionetwork_0.hdf5'
    img_epochs = 10
    aud_epochs = 20
else:
    load_filename = '/home/chandrakanth/learning_experiment/noun_training/Saved_models/noun_model/saved-proxy-audionetwork_' + str(session-1) + '.hdf5'


############################################################
print('Training IMAGES: ')
#Phase1: Train only on images

folder = "Saved_models/noun_model"

filepath = folder+ "/saved-proxy-imagenetwork_" + str(session) + ".hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=True, save_weights_only=True, period=1)
callbacks_list = [checkpoint]

jointnet = JointNet(np.zeros((vec_length, NUM_CLASSES)), False)
model = jointnet.model

#loading weights

model.load_weights(load_filename, by_name=True)

#model.layers[5].layers[1].trainable  =False   #making LSTM untrainable

history = model.fit_generator(generator(df_img_train, 0), verbose=0 ,epochs=img_epochs,steps_per_epoch = len(df_img_train)/BATCH_SIZE, callbacks=callbacks_list, initial_epoch=0,validation_data = generator(df_img_val,0),validation_steps = len(df_img_val)/BATCH_SIZE)

################################################################
print('Training AUDIO: ')

filepath = folder+"/saved-proxy-audionetwork_" + str(session)+ ".hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=True, save_weights_only=True, period=1)
callbacks_list = [checkpoint]

# # Phase2: Train only on audio
jointnet = JointNet(np.zeros((vec_length, NUM_CLASSES)), False)
model = jointnet.model
model.load_weights(folder +'/saved-proxy-imagenetwork_' + str(session) + '.hdf5', by_name=True)    # Choose epoch weights to load

#model.layers[5].layers[1].trainable  =False #making LSTM untrainable


history = model.fit_generator(generator(df_aud_train, 1), verbose=0 ,epochs=aud_epochs, steps_per_epoch = len(df_aud_train)/BATCH_SIZE, callbacks=callbacks_list, initial_epoch=0,validation_data = generator(df_aud_val,1),validation_steps = len(df_aud_val)/BATCH_SIZE)

