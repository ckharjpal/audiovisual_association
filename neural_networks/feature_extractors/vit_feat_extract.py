from PIL import Image
import os
import torch
from transformers import ViTFeatureExtractor, ViTModel
import pickle

def image_features(image ,model,feature_extractor):
  inputs = feature_extractor(images=image, return_tensors="pt")
  outputs = model(**inputs)
  features = outputs['pooler_output']

  return features


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

################
#put the file path and run the code

filename = ''

image = Image.open(filename)
image = image.convert('RGB')
    
feats = image_features(image ,model,feature_extractor)