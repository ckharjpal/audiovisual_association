import numpy as np
from functions.data import im_load
from keras.applications.xception import preprocess_input, Xception
from keras.models import Model
import os

def get_image_model_xception():
    xception_model = Xception(weights='imagenet')
    image_model = Model(inputs=[xception_model.input], outputs=[xception_model.layers[-2].output])
    image_model.trainable = False
    return image_model


def feat_extract_2048(image_filename, image_model):
    image_batch = np.zeros([1,299,299,3])
    image_batch[0] = im_load(image_filename)
    preprocess_input(image_batch)
    image_feat = image_model.predict(image_batch)[0,:]

    return image_feat

#######################################3

image_model = get_image_model_xception()

filename = ''

feats = feat_extract_2048(filename ,image_model)

