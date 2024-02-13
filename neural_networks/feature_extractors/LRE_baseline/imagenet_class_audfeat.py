import os
import configparser as cp
from lre_system import LRESystem as LRE17System
import pickle
import numpy as np
from pdb import set_trace as bp


#############################################

config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
try:
    config.read('lre17_bnf_baseline.cfg')
except:
    raise IOError('Something is wrong with the config file.')
    
print(config['Paths'])

lre17_system = LRE17System(config)

##################################
base_path = '/home/chandrakanth/few_shot_10_class/TTS_data/microsoft_synth_8k'

n_each_class = 15
n_class = 576

audio_length= 36960
feat_length = 229

#labels = open('/home/chandrakanth/few_shot_10_class/classes_10.txt','r').read().split('\n')[:-1]

labels = open('/home/chandrakanth/Audio-Visual-Deep-Multimodal-Networks-master/classes_576.txt','r').read().split('\n')[:-1]

###############################################

audio_feats = {}

for i in range(n_class):

    audio_feats_class = np.zeros([n_each_class, feat_length,80])

    for j in range(n_each_class):

        filename = base_path + '/synthesize-audio_' + labels[i] + '_'+ str(j) + '.wav'
        #filename = base_path + '/' + labels[i] + '_' + str(j) + '.wav'
        

        audio_feats_file = lre17_system.extract_feat_and_apply_sad_then_cmvn(filename)
        audio_feats_file = audio_feats_file.T
        
        audio_len = len(audio_feats_file)

        audio_feats_class[j,:audio_len,:] = audio_feats_file
        
        

    audio_feats[labels[i]] = audio_feats_class
    print(i)


###########################################

#dump into pickle file
with open('/home/chandrakanth/learning_experiment/Data/ASR_feats/audio_features_microsoft.pkl'.format(base_path), 'wb') as file:
    pickle.dump(audio_feats, file)