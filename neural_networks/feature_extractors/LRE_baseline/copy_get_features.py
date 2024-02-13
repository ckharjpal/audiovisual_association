#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:53:55 2019

@author: shreyasr
"""

import os
import configparser as cp
from lre_system import LRESystem as LRE17System
import pickle
from pdb import set_trace as bp
import numpy as np

config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
try:
    config.read('lre17_bnf_baseline.cfg')
except:
    raise IOError('Something is wrong with the config file.')
      
print(config['Paths'])

lre17_system = LRE17System(config)
##################################################

#base_path = '/home/chandrakanth/few_shot_10_class/TTS_data/'
base_path = '/home/chandrakanth/few_shot_10_class/NOUN_dataset/noun_tts/'  #NOUN dataset

#locs = ['microsoft_synth_8k','ibm_synth_8k','google_synth_8k']
locs = ['microsoft_synth','ibm_synth','google_synth']  #For NOUN

pkls = ['microsoft_noun_features.pkl','ibm_noun_features.pkl', 'google_noun_features.pkl']
# pkls = ['google_synth_features_jap.pkl', 'ibm_synth_features_jap.pkl', 'microsoft_synth_features_jap.pkl'] 

seq_length = 229
feat_length = 80

for j, loc in enumerate(locs):
    features = {}
    
    for i, w in enumerate(os.listdir(base_path+loc)):
        if(w[-3:]=='wav'):
            print(i)
            temp_feat = np.zeros([seq_length, feat_length])
            
            feat = lre17_system.extract_feat_and_apply_sad_then_cmvn(base_path+loc+'/'+w)
            feat = feat.T
            
            temp_feat[:len(feat),:] = feat
            #bp()
            features[w] = temp_feat
   
   
    with open('/home/chandrakanth/learning_experiment/Data/noun_audio_feats/' + pkls[j],'wb') as fp:
        pickle.dump(features,fp)
	
