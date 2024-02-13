'''
Code to get image and English audio retrieval accuracies
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pickle
from model import JointNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'


def get_model(filepath):

    jointnet = JointNet(np.zeros((512, NUM_CLASSES)),False)
    
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
    
    print('audio anchor')
    
    for i in range(NUM_CONF):
        print(i)
        cmat = np.zeros((NUM_CLASSES, NUM_CLASSES))
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
                
                cmat[x][y] = np.dot(v1, v2)
        
        folder = 'Confusions/confusion_proxy_audio_anchor/'
        if not os.path.isdir(folder):
            os.system('mkdir '+folder)
        with open('Confusions/confusion_proxy_audio_anchor/confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)

    
    # Get image anchor confusions
    
    print('image anchor')
    for i in range(NUM_CONF):
        print(i)
        cmat = np.zeros((NUM_CLASSES, NUM_CLASSES))
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
                cmat[x][y] = np.dot(v1, v2)

        folder = 'Confusions/confusion_proxy_image_anchor/'
        if not os.path.isdir(folder):
            os.system('mkdir '+folder)
        with open('Confusions/confusion_proxy_image_anchor/confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)


def top_k(cmat, k):
    
    acc = 0
    for x in range(NUM_CLASSES):
        topk_inds = np.argsort(cmat[x,:])[-k:]
        if x in topk_inds:
            acc += 1

    acc = acc/NUM_CLASSES
    return acc

    
def accuracy(folder):
    
    acc_top1 = 0
    acc_top5 = 0
    acc_top10 = 0
    for i in range(NUM_CONF):
        with open(folder + 'confusion_mat_' + str(i) + '.pkl', 'rb') as fp:
            cmat = pickle.load(fp)
        
        acc_top1 += top_k(cmat, 1)
        acc_top5 += top_k(cmat, 5)
        acc_top10 +=top_k(cmat, 10)
        
    acc_top1 = 100*acc_top1/NUM_CONF
    acc_top5 = 100*acc_top5/NUM_CONF
    acc_top10 = 100*acc_top10/NUM_CONF
    
    return acc_top1, acc_top5, acc_top10
  

if __name__=='__main__':

    NUM_CONF = 5
    
    image_test_path = 'Data/image_features_test.pkl'  #put the path of Imagenet test files
    audio_test_path = 'Data/audio_features_test.pkl'
    
    classes = open('/home/chandrakanth/Audio-Visual-Deep-Multimodal-Networks-master/classes_576.txt').read().split('\n')
    classes = classes[:-1]
    NUM_CLASSES = len(classes)

    with open(image_test_path, 'rb') as fp:
        img_data = pickle.load(fp)

    with open(audio_test_path, 'rb') as fp:
        speech_data = pickle.load(fp) 

    filepath = 'Saved_models/saved-proxy-audionetwork.hdf5'    # choose model to load
    
    
    img_transform, aud_transform = get_model(filepath)

    get_confusion(speech_data, img_data, img_transform, aud_transform)
    
    print('Image retrieval accuracy(in %):')
    top1, top5, top10 = accuracy('Confusions/confusion_proxy_audio_anchor/')
    print('Top 1: ' + str(top1))
    print('Top 5: ' + str(top5))
    print('Top 10: ' + str(top10))
    
    print('Audio retrieval accuracy(in %):')
    top1, top5, top10 = accuracy('Confusions/confusion_proxy_image_anchor/')
    print('Top 1: ' + str(top1))
    print('Top 5: ' + str(top5))
    print('Top 10: ' + str(top10))
    
