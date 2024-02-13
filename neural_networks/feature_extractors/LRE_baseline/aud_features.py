import os
import configparser as cp
from lre_system import LRESystem as LRE17System
import pickle
import soundfile as sf
import numpy as np
from pdb import set_trace as bp

config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
try:
    config.read('lre17_bnf_baseline.cfg')
except:
    raise IOError('Something is wrong with the config file.')
    
print(config['Paths'])

lre17_system = LRE17System(config)

##############################

length = 20594
aud = np.zeros(length)
#to generate audio file
#import soundfile as sf
sf.write('test.wav',aud, 16000)


pad_audio = '/home/chandrakanth/Audio-Visual-Deep-Multimodal-Networks-master/Preprocessing/LRE_baseline/test.wav'

audio_feats = lre17_system.extract_feat_and_apply_sad_then_cmvn(pad_audio)
audio_feats = audio_feats.T

#bp()
#print(audio_feats)

print(audio_feats.shape)
