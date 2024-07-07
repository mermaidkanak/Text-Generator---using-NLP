from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer

with open('/content/drive/MyDrive/poem.txt') as story:
  story_data = story.read()

print (story_data)

import re
def clean_text(text):
  text = re.sub(r',', ' ', text)
  text = re.sub(r'\.', ' ', text)
  text = re.sub(r'\"', ' ', text)
  text = re.sub(r'\(',' ', text)
  text = re.sub(r'\)',' ', text)
  text = re.sub(r'\n',' ', text)
  text = re.sub(r'"',' ', text)
  text = re.sub(r'"',' ', text)
  text = re.sub (r'\.',' ', text)
  text = re.sub(r'\'',' ', text)
  text = re.sub(r':',' ', text)
  text = re.sub(r';',' ', text)
  text = re.sub(r'\-',' ', text)
  return text

final = clean_text(story_data)
final_data = final.split('\n')
print(final_data)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_vocab = 100000
tokenizer = Tokenizer(num_words = max_vocab)
tokenizer.fit_on_texts(final_data)
sequences = tokenizer.texts_to_sequences(final_data)

word2idx = tokenizer.word_index
print(len(word2idx))
print(word2idx)
vocab_size = len(word2idx) + 1
print(vocab_size)

input_seq = []
for line in final_data:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_seq = token_list[:i+1]
    input_seq.append(n_gram_seq)

print(input_seq)

max_seq_length = max([len(x) for x in input_seq])
print(max_seq_length)

input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_length, padding='pre'))
print(input_seq)

xs = input_seq[:, :-1]
labels = input_seq[:, -1]
print("xs: ",xs)
print("labels: ",labels)

from tensorflow.keras.utils import to_categorical

ys= to_categorical(labels, num_classes=vocab_size)
print(ys)

from tensorflow.keras.layers import Input, Dense , Embedding, LSTM, Bidirectional, Dropout, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

i = Input(shape=(max_seq_length-1,))
x = Embedding(vocab_size, 124)(i)
x = Dropout(0,2)(x)
x = LSTM(520, return_sequences=True)(x)
x = Bidirectional(layer=LSTM(340,return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
x = Dense(vocab_size, activation='softmax')(x)
model = Model(i, x)

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(xs,ys,epochs=100)

import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'])

def predict_words(seed, no_words):
  for i in range(no_words):
    token_list = tokenizer.texts_to_sequences([seed])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis = 1)

    new_word = ''
    for word, index in tokenizer.word_index.items():
      if predicted == index:
        new_word = word
        break
    seed += ' ' + new_word
  print(seed)

seed_text = 'And the hoof-prints vanish away'
next_words = 10
predict_words(seed_text, next_words)

model.save('poem_generator.h5')
