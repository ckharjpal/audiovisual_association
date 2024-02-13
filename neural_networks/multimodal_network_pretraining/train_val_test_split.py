import pickle
import random
import numpy as np

##################################################
#loading data

with open('/home/chandrakanth/few_shot_10_class/image_features_imagenet_train.pkl', 'rb') as fp:
    image_data = pickle.load(fp)

with open('/home/chandrakanth/learning_experiment/Data/ASR_feats/audio_features_google.pkl', 'rb') as fp:
    audio_data_google = pickle.load(fp)

with open('/home/chandrakanth/learning_experiment/Data/ASR_feats/audio_features_ibm.pkl', 'rb') as fp:
    audio_data_ibm = pickle.load(fp)

with open('/home/chandrakanth/learning_experiment/Data/ASR_feats/audio_features_microsoft.pkl', 'rb') as fp:
    audio_data_microsoft = pickle.load(fp)

print('data loaded')

###################################################
n_class = 576

img_feat_size  =768
aud_feat_size = 80  #80 for ASR and 768 for wav2vec

aud_max_len = 229    #change depending on audio set used

#####################################################
#image data

n_train_img =24
n_test_img = 8
n_val_img = 8

n_total_img = 200


img_train  ={}
img_test = {}
img_val = {}

img_keys = list(image_data.keys())

labels = open('/home/chandrakanth/Audio-Visual-Deep-Multimodal-Networks-master/classes_576.txt','r').read().split('\n')[:-1]
base_path = '/home/chandrakanth/few_shot_10_class/training'

imagenetmap = open('/home/chandrakanth/New-Experiment/imagenet_map.txt','r').read().split('\n')
imagenetmap = {i.split()[2].lower():i.split()[0] for i in imagenetmap if i.split()[2].lower() in labels}
imagemaprev = {i:j for j,i in imagenetmap.items()}


for i in img_keys:

    temp_img = image_data[i]
    temp_img = temp_img.numpy()

    n_list = [k for k in range(n_total_img)]
    n_list = np.array(n_list)

    random.shuffle(n_list)

    img_train_temp = [temp_img[n_list[j]] for j in range(n_train_img)]

    img_test_temp = [temp_img[n_list[j]] for j in range(n_train_img, n_train_img+ n_test_img)]

    img_val_temp = [temp_img[n_list[j]] for j in range(n_train_img+n_test_img , n_train_img+ n_test_img+n_val_img)]

    label_class = imagemaprev[i]

    img_train_temp = np.array(img_train_temp)
    img_test_temp = np.array(img_test_temp)
    img_val_temp = np.array(img_val_temp)

    img_train[label_class] = img_train_temp
    img_test[label_class] = img_test_temp
    img_val[label_class] = img_val_temp

##################################################
#merging of audio data into a single dictionary

aud_keys = list(audio_data_google.keys())

n_total_aud = 14+3+15 #n_google+n_ibm + n_microsoft
audio_data = {}

for i in aud_keys:

    aud_google = audio_data_google[i]
    aud_ibm = audio_data_ibm[i]
    aud_microsoft = audio_data_microsoft[i]

    temp_aud = np.vstack((aud_google, aud_ibm, aud_microsoft))

    audio_data[i] = temp_aud

del audio_data_google, audio_data_ibm, audio_data_microsoft

#################################################

#audio data

n_train_aud =24
n_test_aud = 8
n_val_aud = 2

n_total_aud = 32


aud_train  ={}
aud_test = {}
aud_val = {}

#aud_keys = list(audio_data.keys())


for i in aud_keys:

    temp_aud = audio_data[i]
    #temp_aud = temp_aud.numpy()

    n_list = [k for k in range(n_total_aud)]
    n_list = np.array(n_list)

    random.shuffle(n_list)

    aud_train_temp = [temp_aud[n_list[j],:,:] for j in range(n_train_aud)]

    aud_test_temp = [temp_aud[n_list[j],:,:] for j in range(n_train_aud, n_train_aud+ n_test_aud)]

    aud_val_temp = [temp_aud[n_list[j],:,:] for j in range(n_train_aud+n_test_aud , n_train_aud+ n_test_aud+n_val_aud)]

    aud_train_temp = np.array(aud_train_temp)
    aud_test_temp = np.array(aud_test_temp)
    aud_val_temp = np.array(aud_val_temp)

    label_class = i

    aud_train[label_class] = aud_train_temp
    aud_test[label_class] = aud_test_temp
    aud_val[label_class] = aud_val_temp

######################################################################

#writing to pickle files

with open('../Data/ASR_feats/image_features_24_train.pkl'.format(base_path), 'wb') as file:
    pickle.dump(img_train, file)

with open('../Data/ASR_feats/image_features_24_val.pkl'.format(base_path), 'wb') as file:
    pickle.dump(img_val, file)

with open('../Data/ASR_feats/image_features_24_test.pkl'.format(base_path), 'wb') as file:
    pickle.dump(img_test, file)

with open('../Data/ASR_feats/audio_features_train.pkl'.format(base_path), 'wb') as file:
    pickle.dump(aud_train, file)

with open('Data/all_audio_features_val.pkl'.format(base_path), 'wb') as file:
   pickle.dump(aud_val, file)

with open('../Data/ASR_feats/audio_features_test.pkl'.format(base_path), 'wb') as file:
    pickle.dump(aud_test, file)

print('Done')