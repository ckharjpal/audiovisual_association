Folders

feature_extractors contains code for extracting features for both audios and images.

Multimodal_network_pretraining contains code to train the multimodal audiovisual network with 576 classes of ImageNet. It should be done after downloading ImageNet images and using TTS to generate audios. The data is too large to upload and can be easily found online.

NOUN_training contains code to train the part of multimodal network with NOUN images and audios which are included in human_behavior_experiments/stimuli folder. This should be run after extracting features. It also contain implementation of EWC in the audiovisual network.