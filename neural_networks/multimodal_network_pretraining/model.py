import keras.backend as K
from keras.models import Model,  Sequential
from keras.layers import Lambda, Layer, Dense, Input, Dropout, Multiply, Add, Masking, LSTM, BatchNormalization, Activation
import tensorflow as tf
from pdb import set_trace as bp
# from keras_layer_normalization import LayerNormalization


class MyLayer(Layer):

    def __init__(self, output_dim, init, trainable, **kwargs):
        self.output_dim = output_dim
        self.init = init
        self.trainable = trainable
        # print("***********************{},{},{}".format(self.output_dim,self.init,self.trainable))
        super(MyLayer, self).__init__(**kwargs)


    def my_init(self, shape, dtype=None):
        return K.variable(self.init)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.my_init,
                                      trainable=self.trainable)   

        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return 20 * K.dot(x, K.l2_normalize(self.kernel, axis=0))       # Temperature factor of 20

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class JointNet():

    def __init__(self, proxy_init, trainable):

        self.init = proxy_init
        self.trainable = trainable
        self.audio_submodel = self.audio_submodel()
        self.image_submodel = self.image_submodel()
        self.model = self.joint_model()
        

    def identity_loss(self, y_true, y_pred):

        return K.mean(y_pred)


    def bpr_proxy_loss(self, X):

        distances, class_mask, class_mask_bar = X


        # COSINE SIMILARITY
        d_pos = K.exp(K.sum(distances * class_mask, axis=-1, keepdims=True))
        d_neg = K.sum(K.exp(distances * class_mask_bar), axis=-1, keepdims=True) - 1

        loss = K.log(tf.divide(d_neg, d_pos))

        return loss


    def audio_submodel(self):

        input_size = 80
        hidden_size = 128
        
        model = Sequential()
        # audio submodel
        model.add(Masking(mask_value=0.0, input_shape=(None, input_size), name='masking_1'))		
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size), name='lstm_1', trainable=True))
        model.add(LSTM(hidden_size, return_sequences=False, input_shape=(None, hidden_size), name='lstm_2', trainable=True))
        model.add(Dense(2048, name='dense_2048', trainable=True))
        model.add(BatchNormalization(name='batch_normalization_2048', trainable=True))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        #uncomment next line and put audio submodel trained

        #model.load_weights('Saved_models/model_audio_subnetwork.h5', by_name=True)
        

        inp = Input((None, 80))
        op1 = model(inp)
        op2 = Dense(512, activation='tanh', input_shape=(2048, ), name='dense_aud1', trainable=True)(op1)
        
        model = Model(inputs=inp, outputs=op2, name='sequential_1')

        return model

    
    def image_submodel(self):

        inp = Input((768, ))
        op1 = Dropout(0.5)(inp)
        op2 = Dense(512, activation='tanh', name='dense_img1', trainable=True)(op1)

        model = Model(inputs=inp, outputs=op2, name='sequential_2')
        
        return model


    def joint_model(self):

        NUM_CLASSES = 576
        aud_feat_size = 80
        img_feat_size  = 768

        grounding = Input((1, ), name='grounding')
        grounding_bar = Input((1, ), name='grounding_bar')
        anchor_aud = Input((None, aud_feat_size), name='anchor_aud')
        anchor_img = Input((img_feat_size, ), name='anchor_img')
        class_mask = Input((NUM_CLASSES, ), name='class_mask')
        class_mask_bar = Input((NUM_CLASSES, ), name='class_mask_bar')
        

        anchor_aud_latent = self.audio_submodel(anchor_aud)
        anchor_img_latent = self.image_submodel(anchor_img)
     

        anchor = Add()([Multiply()([grounding_bar, anchor_img_latent]), Multiply()([grounding, anchor_aud_latent])])     
        anchor_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))(anchor)


        proxy_mat = MyLayer(NUM_CLASSES, self.init, self.trainable, name='my_layer_1')
        distances = proxy_mat(anchor_norm)
        
        
        loss_cosine = Lambda(self.bpr_proxy_loss,output_shape=(1,),name='loss')([distances, class_mask, class_mask_bar])
        model = Model(inputs=[grounding, grounding_bar, anchor_aud, anchor_img, class_mask, class_mask_bar],outputs=loss_cosine)
        model.compile(loss=self.identity_loss, optimizer=tf.keras.optimizers.Adam(lr=0.1))
        
        return model
