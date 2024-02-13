import os
import configparser as cp
from lre_system import LRESystem as LRE17System


config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
try:
    config.read('lre17_bnf_baseline.cfg')
except:
    raise IOError('Something is wrong with the config file.')
    
print(config['Paths'])
#exit()
lre17_system = LRE17System(config)
base_path = '/home/chandrakanth/few_shot_10_class/TTS_data/google_synth_8k/synthesize-audio_coyote_2.wav'

features = {}

feat = lre17_system.extract_feat_and_apply_sad_then_cmvn(base_path)
