from django.shortcuts import render
from .ml_utils import load_model, load_tokenizer

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_GET
import json

import json
import os
from pathlib import Path
# import the necessary libraries for ml
import tensorflow as tf
# import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
# from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
# importing stopwords
import nltk
nltk.download('stopwords')

# Loading DistilBERT Tokenizer and the DistilBERT model
dbert_tokenizer = load_tokenizer()
dbert_model = load_model()

print(dbert_model.summary())


# processing and cleaning functions
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def clean_stopwords_shortwords(w):
    stopwords_list = stopwords.words('english')
    words = w.split()
    clean_words = [word for word in words if (
        word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words)


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = clean_stopwords_shortwords(w)
    w = re.sub(r'@\w+', '', w)
    return w


# Set the maximum length of the input sentences
max_len = 32
num_classes = 6  # number of classes in our dataset
log_dir = 'dbert_model'
# script_dir = os.path.dirname(__file__)  # Get the directory of the script
# model_save_path = os.path.join(script_dir, 'dbert_model.h5')
# print(model_save_path)

# Get the directory of the script
script_dir = Path(__file__).resolve().parent

# Define the relative path to the model
model_relative_path = '../../dbert_model.h5'

# Combine the script directory with the relative path to get the absolute path
model_save_path = script_dir / model_relative_path

print(model_save_path)

# now we try to predict the label of a random sentence from user


def create_model():
    inps = Input(shape=(max_len,), dtype='int64')
    masks = Input(shape=(max_len,), dtype='int64')
    dbert_layer = dbert_model(inps, attention_mask=masks)[0][:, 0, :]
    dense = Dense(512, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01))(dbert_layer)
    dropout = Dropout(0.5)(dense)
    pred = Dense(num_classes, activation='softmax',
                 kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = tf.keras.Model(inputs=[inps, masks], outputs=pred)
    print(model.summary())
    return model


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

trained_model = create_model()
trained_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
trained_model.load_weights(model_save_path)  # loading the model


def predict(sentence):
    # we first preprocess the sentence
    sentence = preprocess_sentence(sentence)
    # then we do some tokenization on the sentence
    dbert_inps = dbert_tokenizer.encode_plus(
        sentence, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, truncation=True)
    # then we convert to numpy array
    id_inp = np.asarray(dbert_inps['input_ids'])
    mask_inp = np.asarray(dbert_inps['attention_mask'])
    # and predict the label using the trained model
    pred = trained_model.predict(
        [id_inp.reshape(1, -1), mask_inp.reshape(1, -1)])
    # and then get the label with the highest probability
    pred_label = np.argmax(pred, axis=1)
    # we then return it
    label_class_dict = {0: 'sadness', 1: 'happy',
                        2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

    return label_class_dict[pred_label[0]]


@csrf_exempt  # decorator to exempt from CSRF protection
@require_POST
def predict_sentiment(request):
    try:
        if request.method == 'POST':
            data = json.loads(request.body.decode('utf-8'))
            input_text = data.get("text", "")
            print(input_text)
            print("Before prediction")
            prediction = predict(input_text)
            print("After prediction:", prediction)
            print("prediction below => ")
            print(prediction)
            return JsonResponse({"prediction": prediction})
        else:
            return JsonResponse({"error": "Invalid HTTP method. Use POST."})
    except Exception as e:
        return JsonResponse({"error": str(e)})
