import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import pickle
from pdb import set_trace as bp

#CUDA devices enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

# load model and processor 
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)

embedding_dim = model.config.to_dict()['hidden_size']
model.cuda()

#########################

def audio_feat(audio_file):

  with torch.no_grad():
    inputs = processor(audio_file, sampling_rate=16000, return_tensors="pt")
    for x in inputs.keys():
        inputs[x]= inputs[x].cuda()
    outputs = model(**inputs)

  features = outputs['hidden_states']

  return features[-1]

##################################

#uncomment and put the file path 

#audio_filename = 

#(aud_sig, sr) = librosa.load(filename,sr=16000)

#audio_features = audio_feat(aud_sig)

