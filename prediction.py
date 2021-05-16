import nltk
import pandas as pd
import re
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
y = np.array(Y)

## tokenize the word to create embedding
vocab_size = 2000
tk = keras.preprocessing.text.Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tk.fit_on_texts(x)
x_new = tk.texts_to_sequences(x) ## use this while getting inputs
word_index = tk.word_index  # index of unique words


from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
max_len = 50       #length of sequence
batch_size = 32
epochs = 5
max_features = 100
x_new = pad_sequences(x_new, maxlen=max_len, padding='pre')
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)


# load json and create model
json_file = open('./model_disaster_tuned.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model_disaster_tuned.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer=keras.optimizers.Adam(0.0001),
                 loss=keras.losses.binary_crossentropy,
                 metrics=['accuracy'])


## so that we don't have to compile every time
loaded_model.save('final_model.h5')
#loaded_model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=32)
##loaded_model.evaluate(x_test,y_test)





