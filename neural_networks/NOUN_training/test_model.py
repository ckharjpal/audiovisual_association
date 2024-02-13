'''
Code to get image and English audio retrieval accuracies
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pickle
import sys
from model_practice import JointNet
from pdb import set_trace as bp
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

def get_model(filepath):

    jointnet = JointNet(np.zeros((576, NUM_CLASSES)),False)
    # jointnet = JointNet(False)
    model = jointnet.model
    model.load_weights(filepath, by_name=True)
    aud_transform = jointnet.audio_submodel
    img_transform = jointnet.image_submodel
    
    return img_transform, aud_transform

    
def get_confusion(speech_data, img_data_test, img_transform, aud_transform):

    aud_features = []
    for c in classes:
        for s in speech_data[c]:
            aud_features.append(s)

    aud_latent = aud_transform.predict(np.array(aud_features))
   
    img_features = []
    for c in classes:
        for s in range(len(img_data_test[c])):
            img_features.append(img_data_test[c][s])
            
    img_latent = img_transform.predict(np.array(img_features))
 
    # Get audio anchor confusions

    num_speakers = len(speech_data[list(speech_data)[0]])
    
    cmat_audio = np.zeros((NUM_CONF,NUM_CLASSES, NUM_CLASSES))
    cmat_image = np.zeros((NUM_CONF,NUM_CLASSES, NUM_CLASSES))
    
    print('audio anchor')
    for i in range(NUM_CONF):
        
        for x, ca in enumerate(classes):
            spk_ind = np.random.randint(num_speakers)
            v1 = aud_latent[x*num_speakers + spk_ind]
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            ind_done = 0
            for y, ci in enumerate(classes):
                num_images = len(img_data_test[ci])
                img_ind = np.random.randint(num_images)
                v2 = img_latent[ind_done + img_ind]            
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                ind_done+=num_images
                
                cmat_audio[i][x][y] = np.dot(v1, v2)
        
    # Get image anchor confusions
    
    print('image anchor')
    for i in range(NUM_CONF):
        
        ind_done = 0
        for x, ci in enumerate(classes):
            num_images = len(img_data_test[ci])
            img_ind = np.random.randint(num_images)
            v1 = img_latent[ind_done+ img_ind]
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            ind_done+=num_images
            for y, ca in enumerate(classes):
                spk_ind = np.random.randint(num_speakers)
                v2 = aud_latent[y*num_speakers + spk_ind]
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                
                cmat_image[i][x][y] = np.dot(v1, v2)
                
    return cmat_image, cmat_audio


def top_k(cmat, k):
    
    acc = 0
    for x in range(NUM_CLASSES):
        #bp()
        topk_inds = np.argsort(cmat[x,:])[-k:]
        if x in topk_inds:
            acc += 1

    acc = acc/NUM_CLASSES
    return acc

    
def accuracy(cmat):
    
    acc_top1 = 0
    acc_top5 = 0
    acc_top10 = 0
    for i in range(NUM_CONF):
        
        cmat_k = cmat[i]
       
        acc_top1 += top_k(cmat_k, 1)
        acc_top5 += top_k(cmat_k, 5)
        acc_top10 +=top_k(cmat_k, 10)
        
    acc_top1 = 100*acc_top1/NUM_CONF
    acc_top5 = 100*acc_top5/NUM_CONF
    acc_top10 = 100*acc_top10/NUM_CONF
    
    return acc_top1, acc_top5, acc_top10
  

if __name__=='__main__':

    NUM_CONF = 8
    session = int(sys.argv[1])
    net_session = int(sys.argv[2])
    
    image_test_path = '/home/chandrakanth/learning_experiment/Data/NOUN_train_test_split/image_features_test.pkl'
    audio_test_path = '/home/chandrakanth/learning_experiment/Data/NOUN_train_test_split/audio_features_test.pkl'
    

    with open(image_test_path, 'rb') as fp:
        img_data_session = pickle.load(fp)

    with open(audio_test_path, 'rb') as fp:
        speech_data_session = pickle.load(fp) 
        
    img_data_test = img_data_session[session]
    speech_data = speech_data_session[session]
        
    classes = list(img_data_test.keys())
    NUM_CLASSES = len(classes)
    
    net_filename = ''

 
    img_transform, aud_transform = get_model(net_filename)

    cmat_image = np.zeros([NUM_CONF, NUM_CLASSES, NUM_CLASSES])
    cmat_audio = np.zeros([NUM_CONF, NUM_CLASSES, NUM_CLASSES])

    cmat_image, cmat_audio =get_confusion(speech_data, img_data_test, img_transform, aud_transform)
    
        
    print('Image retrieval accuracy(in %):')
    top1, top5, top10 = accuracy(cmat_audio)
    print('Top 1: ' + str(top1))
    print('Top 5: ' + str(top5))
    print('Top 10: ' + str(top10))
    
    print('Audio retrieval accuracy(in %):')
    top1, top5, top10 = accuracy(cmat_image)
    print('Top 1: ' + str(top1))
    print('Top 5: ' + str(top5))
    print('Top 10: ' + str(top10))
    
