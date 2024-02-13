import matplotlib
matplotlib.use('Agg')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model_ewc_modified import JointNet
import tensorflow as tf
import keras.backend as K
from copy import deepcopy
import numpy as np
import os
import pickle
from keras.callbacks import ModelCheckpoint
import time
import sys
start = time.time()
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'


#################################################

def fisher_matrix_joint(model, df_img, df_aud, samples):
   
    img_inputs = df_img
    aud_inputs = df_aud
    
    weights = model.model.trainable_weights
    
    variance = [tf.zeros_like(tensor) for tensor in weights]

    for _ in range(samples):
        # Select a random element from the dataset.
        index = np.random.randint(len(img_inputs))
        data_img = img_inputs[index,0]
        data_aud = aud_inputs[index,0]

        # When extracting from the array we lost a dimension so put it back.
        data_img = tf.expand_dims(data_img, axis=0)
        data_aud = tf.expand_dims(data_aud, axis=0)

        # Collect gradients.
        with tf.GradientTape() as tape:
            image_rep = model.image_submodel(data_img)
            audio_rep = model.audio_submodel(data_aud)
            image_rep = tf.squeeze(image_rep, axis =0)
            audio_rep = tf.squeeze(audio_rep, axis=0)
            
            output = tf.tensordot(image_rep, audio_rep, 1)
            

        gradients = tape.gradient(output, weights)
        
        
        for i in range(len(variance)-1):
            variance[i] = variance[i] + gradients[i]**2
            

    fisher_diagonal = [tensor / samples for tensor in variance]
    return fisher_diagonal

def compile_model(model, learning_rate, lam, fisher_diagonal, optimal_weights):
    def custom_loss(y_true, y_pred):
        loss = K.mean(y_pred)
        
        current = model.trainable_weights
        
        loss_ewc =0
        
        for f, c, o in zip(fisher_diagonal, current, optimal_weights):
            loss_ewc += tf.reduce_sum(f * ((c - o) ** 2))
        
        loss = loss + loss_ewc*(lam/2)   

        return loss

    model.compile(
        loss=custom_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    

def ewc_loss(lam, model, fisher_diagonal):
    
    optimal_weights = deepcopy(model.trainable_weights)

    def loss_fn(y_pred, y_true):
        # We're computing:
        # sum [(lambda / 2) * F * (current weights - optimal weights)^2]
        loss = K.mean(y_pred)
        current = model.trainable_weights
        
        for f, c, o in zip(fisher_diagonal, current, optimal_weights):
            loss += tf.reduce_sum(f * ((c - o) ** 2))

        return loss * (lam / 2)

    return loss_fn


#####################################

vec_length = 576

#session = 0   #goes from 0 to 5
session = int(sys.argv[1])
lam = 100

img_epochs = 6
aud_epochs = 10
#############################################

path = '/home/chandrakanth/learning_experiment/Data/NOUN_train_test_split/'

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

img_val_ewc = img_val_session[session-1]
aud_val_ewc= audio_val_session[session-1]

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
	df = np.array(df)
	return df

df_img_train = createdf(img_train)
df_img_val = createdf(img_val)
df_aud_train = createdf(audio_train)
df_aud_val = createdf(audio_val)

df_img_val_ewc = createdf(img_val_ewc)
df_aud_val_ewc = createdf(aud_val_ewc)


BATCH_SIZE = 5
#################################################################
if session ==1:
    load_filename = '/home/chandrakanth/learning_experiment/noun_training/Saved_models/noun_model/saved-proxy-audionetwork_0.hdf5'
else:
    load_filename = '/home/chandrakanth/learning_experiment/noun_training/Saved_models/noun_model/saved-proxy-audionetwork_ewc_' + str(session-1) + '.hdf5'


############################################################
print('Training IMAGES: ')
#Phase1: Train only on images

folder = "Saved_models/noun_model"

filepath = folder+ "/saved-proxy-imagenetwork_ewc_" + str(session) + ".hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=True, save_weights_only=True, period=1)
callbacks_list = [checkpoint]

jointnet = JointNet(np.zeros((vec_length, NUM_CLASSES)), False)
model = jointnet.model

model.load_weights(load_filename, by_name=True)

##############################################################


#estimating fisher diagonal matrix


fisher_diagonal = fisher_matrix_joint(jointnet, df_img_val, df_aud_val, 30)

optimal_weights = deepcopy(model.trainable_weights)

compile_model(model, 0.01, lam , fisher_diagonal, optimal_weights)

##############################################

history = model.fit_generator(generator(df_img_train, 0),verbose=0, epochs=img_epochs,steps_per_epoch = len(df_img_train)/BATCH_SIZE, callbacks=callbacks_list, initial_epoch=0,validation_data = generator(df_img_val,0),validation_steps = len(df_img_val)/BATCH_SIZE)

################################################################
print('Training AUDIO: ')

filepath = folder+"/saved-proxy-audionetwork_ewc_" + str(session)+ ".hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=True, save_weights_only=True, period=1)
callbacks_list = [checkpoint]


# # Phase2: Train only on audio
jointnet = JointNet(np.zeros((vec_length, NUM_CLASSES)), False)
model = jointnet.model
model.load_weights(folder +'/saved-proxy-imagenetwork_ewc_' + str(session) + '.hdf5', by_name=True)    # Choose epoch weights to load

############################################
fisher_diagonal = fisher_matrix_joint(jointnet, df_img_val, df_aud_val, 30)

optimal_weights = deepcopy(model.trainable_weights)

compile_model(model, 0.01, lam , fisher_diagonal, optimal_weights)

history = model.fit_generator(generator(df_aud_train, 1), verbose=0, epochs=aud_epochs, steps_per_epoch = len(df_aud_train)/BATCH_SIZE, callbacks=callbacks_list, initial_epoch=0,validation_data = generator(df_aud_val,1),validation_steps = len(df_aud_val)/BATCH_SIZE)

