import os
os.environ['PYTHONHASHSEED']= '15'
os.environ['TF_CUDNN_DETERMINISTIC']= '1'

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import random as python_random
import regex as re
import pickle
from gensim.models.fasttext import FastText
from keras.models import load_model

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

np.random.seed(15)
python_random.seed(15)
tf.random.set_seed(15)

MAX_LEN = 698

@st.cache_resource
def load_lstm_cnn():
    model = load_model('model/lstm-cnn')
    return model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def classify(data):
    predictions = model.predict(data)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    return binary_predictions

def text_preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\n', ' ', text)
    text = stemmer.stem(text)
    text = stopword.remove(text)
    return text

with st.spinner():
    model = load_lstm_cnn()
st.title('Klasifikasi Konten Eksplisit pada Lirik Lagu Berbahasa Indonesia Menggunakan Metode LSTM-CNN')
model_option = st.selectbox(
    'Model yang digunakan:',
    ('v1.0.1 (Imbalanced dataset)', 'v1.0.2 (Balanced dataset)'))
lirik = st.text_area('Masukkan lirik lagu yang ingin diklasifikasikan:', height=200)
tombol = st.button('Klasifikasikan')
if tombol:
    if lirik != '':
        lirik2 = text_preprocess(lirik)
        tokenizer = load_tokenizer()
        sequences = tokenizer.texts_to_sequences([lirik2])
        padded = pad_sequences(sequences, maxlen=MAX_LEN)
        padded = np.array(padded)
        hasil = classify(padded)
        if hasil == 1:
            label_hasil = "Eksplisit [1]"
        else:
            label_hasil = "Noneksplisit [0]"
        st.info("Hasil klasifikasi adalah: " + label_hasil)
    else:
        st.warning('Silakan masukkan lirik lagu terlebih dahulu.')
