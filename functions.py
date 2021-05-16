import nltk
import pandas as pd

import keras
import numpy as np
stop = set(nltk.corpus.stopwords.words('english'))
from keras.models import model_from_json

df = pd.read_csv('train.csv')
df_new = df.iloc[:,3:]

X = df_new.text
Y= df_new.target

x_list = []   ## removing stopwords
for i in X:
       
    wor = i.split()
    filtered_words = [w for w in wor if w not in stop]
    i = filtered_words
    i= ' '.join(i)
    x_list.append(i)
    
x = np.array(x_list)


## tokenize the word to create embedding
vocab_size = 2000
tk = keras.preprocessing.text.Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tk.fit_on_texts(x)
from keras.preprocessing.sequence import pad_sequences
max_len = 50  

loaded_model = keras.models.load_model('./final_model.h5')


def preprocess(input1):
    list1 = []
    wor = input1.split()
    filtered_words = [w for w in wor if w not in stop]
    i = filtered_words
    i = ' '.join(i)
    list1.append(i)
    x_1 = np.array(list1)
    #print(i)

    ## tokenize the word to create embedding
    
    tk.fit_on_texts(x_1)
    x_new = tk.texts_to_sequences(x_1)
    
    x_new_1 = pad_sequences(x_new, maxlen=max_len, padding='pre')
    #print(x_new_1)
    y = loaded_model.predict(x_new_1)
    if y>0.5:
        return 1
    else:
        return 0