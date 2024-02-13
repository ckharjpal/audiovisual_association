import librosa
import os

#gives the max length of audio file present in a directory

base_path = '/home/chandrakanth/few_shot_10_class/TTS_data/microsoft_synth_eng_exp'

files = sorted([item for item in os.listdir(base_path)])

max_length = 0


for i in range(len(files)):
    
    aud_file = base_path + '/' + files[i]
    
    (sig, sr) = librosa.load(aud_file,sr=16000)
    
    length = len(sig)
    
    if length >max_length:
        max_length = length

print(max_length)