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


config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
try:
    config.read('lre17_bnf_baseline.cfg')
except:
    raise IOError('Something is wrong with the config file.')
    
print(config['Paths'])
#exit()
lre17_system = LRE17System(config)

#base_path = '/home/chandrakanth/few_shot_10_class/TTS_data/'
base_path = '/home/chandrakanth/few_shot_10_class/NOUN_dataset/noun_tts'  #NOUN dataset

locs = ['microsoft_synth','ibm_synth','google_synth']

# locs = ['google_synth_jap', 'ibm_synth_jap', 'microsoft_synth_jap']
pkls = ['microsoft_noun_features.pkl','ibm_noun_features.pkl', 'google_noun_features.pkl']
# pkls = ['google_synth_features_jap.pkl', 'ibm_synth_features_jap.pkl', 'microsoft_synth_features_jap.pkl'] 

for j, loc in enumerate(locs):
    features = {}
    if loc=='microsoft_synth_8k':
       print('microsoft_synth_eng_exp')
       for i, w in enumerate(os.listdir(base_path+'microsoft_synth_eng_exp')):
           if(w[-3:]=='wav'):
               print(i)
               feat = lre17_system.extract_feat_and_apply_sad_then_cmvn(base_path+'microsoft_synth_eng_exp'+'/'+w)
               features[w] = feat.T
    # if loc[-3:]=='jap':
    #     print(loc+'_exp')
    # for i, w in enumerate(os.listdir(base_path+loc+'_exp')):
    #     if(w[-3:]=='wav'):
    #         print(i)
    #         feat = lre17_system.extract_feat_and_apply_sad_then_cmvn(base_path+loc+'_exp/'+w)
    #         features[w] = feat.T
#    print(loc)
#    for i, w in enumerate(os.listdir(base_path+loc)):
#        if(w[-3:]=='wav'):
#            print(i)
#            feat = lre17_system.extract_feat_and_apply_sad_then_cmvn(base_path+loc+'/'+w)
#            features[w] = feat.T
    with open('../../Data/' + pkls[j],'wb') as fp:
        pickle.dump(features,fp)
	
