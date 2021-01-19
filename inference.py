#remove warnings from displaying in output 
import warnings 
warnings.filterwarnings('ignore')

# importing the required libraries 
import os 
import re 
import nltk
import cv2
import time
import datetime
import pickle
nltk.download('stopwords')
nltk.download('punkt')
from bs4 import BeautifulSoup
from PIL import Image
from skimage.transform import resize
from nltk.corpus import stopwords
from nltk import word_tokenize
from os import listdir
from os import path
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.model_selection import train_test_split
import nltk.translate.bleu_score as bleu
from google.colab.patches import cv2_imshow
from PIL import Image
import xml.etree.ElementTree as ET


import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.keras.backend.clear_session()
from tensorflow.keras.models import Model
from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras import layers 
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPool2D,Activation,Dropout
from tensorflow.keras.layers import LSTM,Embedding,Flatten,BatchNormalization,ReLU
from tensorflow.keras.layers import Softmax,RNN,Reshape,concatenate,TimeDistributed


def get_image_feature_extract_model(img_width,img_height):
    #https://github.com/brucechou1983/CheXNet-Keras
    # weigths : https://drive.google.com/file/d/19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b/view 

    model = tf.keras.applications.DenseNet121(weights=None,classes = 14,input_shape=(int(img_width),int(img_height),3))

    # classes is 14 because its trained on 14 classes classification (multi class classification)
    model.load_weights('/content/drive/My Drive/brucechou1983_CheXNet_Keras_0.3.0_weights.h5')

    image_features_extract_model = Model(inputs=model.input,outputs=model.layers[-3].output)

    return image_features_extract_model

# seq to seq assignment .
class Encoder(tf.keras.Model):
    
    def __init__(self, embedding_dim,feature_extracter):
        super(Encoder, self).__init__()
        self.image_features_extract_model = feature_extracter
    
    def call(self, x):
        x = self.image_features_extract_model(x)
        x = tf.reshape(x,(x.shape[0], -1, x.shape[3]))
        return x
    
#https://www.tensorflow.org/tutorials/text/nmt_with_attention
class attention_concat(tf.keras.layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.units=units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self,decoder_hidden_state,encoder_output_states):
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, 1)
        similarities = self.V(tf.nn.tanh(self.W1(decoder_hidden_state) + self.W2(encoder_output_states)))
        attention_weights = tf.nn.softmax(similarities, axis=1)
        context_vector = attention_weights * encoder_output_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
# https://sezazqureshi.medium.com/chest-x-ray-medical-report-generation-using-deep-learning-bf39cc487b88
# https://www.tensorflow.org/tutorials/text/image_captioning


class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units):
        super(Decoder, self).__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.attention = attention_concat(self.dec_units)
        self.input_length = input_length

        self.embedding = Embedding(input_dim = self.out_vocab_size, output_dim = self.embedding_dim, input_length = self.input_length,
                           mask_zero=True, name="embedding_layer_decoder",trainable=False)
        
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, name="Decoder_LSTM",dropout=0.1,recurrent_dropout=0.1)
    
        self.fc = Dense(self.out_vocab_size)

    @tf.function
    def onestep_decoder(self,input_to_decoder, state_h, encoder_output , state_c):
        '''one step decoder is called for every timestamp'''
        target_embedd  = self.embedding(input_to_decoder)

        context_vector,attention_weights = self.attention(state_h,encoder_output)

        context_vector = tf.expand_dims(context_vector,1)

        concat = tf.keras.layers.Concatenate(axis=-1)([target_embedd,context_vector])

        concat = tf.nn.dropout(concat, 0.1)

        lstm_out, state_h, state_c = self.lstm(concat,initial_state=(state_h,state_c))

        lstm_out = tf.reshape(lstm_out,(-1,lstm_out.shape[2]))

        output = self.fc(lstm_out)

        return output , state_h , attention_weights , state_c

    def call(self, input_to_decoder,decoder_hidden_state,encoder_output,decoder_cell_state):
        all_outputs = tf.TensorArray(tf.float32 , size=input_to_decoder.shape[1] , name = 'output_arrays')
        for timestep in range(input_to_decoder.shape[1]):
            output , decoder_hidden_state ,attn_weights, decoder_cell_state  = self.onestep_decoder(input_to_decoder[:,timestep:timestep+1] ,
                                                            decoder_hidden_state ,encoder_output,decoder_cell_state)
            
            all_outputs = all_outputs.write(timestep,output)
        all_outputs = tf.transpose(all_outputs.stack() , [1,0,2])
        return all_outputs

    def reset_hidden_state(self, batch_size):
      return tf.random.normal((batch_size, self.dec_units))

    def reset_cell_state(self,batch_size):
      return tf.random.normal((batch_size,self.dec_units))
        
def grader_decoder(decoder):
    input_len_dec=123
    units=512
    target_vocab_size=1487
    target_sentences=tf.random.uniform(shape=(1,input_len_dec),maxval=10,minval=0,dtype=tf.int32)
    encoder_output=tf.random.uniform(shape=[1,49,2048])
    state_h=tf.random.uniform(shape=[1,units])
    state_c=tf.random.uniform(shape=[1,units])
    #decoder=Decoder(out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units)
    output=decoder(target_sentences,state_h,encoder_output,state_c)
    assert(output.shape==(1,input_len_dec,target_vocab_size))
    return True
  



