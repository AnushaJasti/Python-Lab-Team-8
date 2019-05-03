import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import backend as K
import re

K.set_image_dim_ordering('th')
import os
import pandas as pd
data = pd.read_csv('sentrain.csv')
test=pd.read_csv('sentest.csv')
# Keeping only the neccessary columns
data = data[['phrase','sentiment']]

data['phrase'] = data['phrase'].apply(lambda x: x.lower())
data['phrase'] = data['phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['phrase'].values)
X = tokenizer.texts_to_sequences(data['phrase'].values)
print(X)
X = pad_sequences(X)
print(X)
embed_dim = 128
lstm_out = 196
print(X.shape)
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(
        Conv1D(128,(5), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv1D(128,(5), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    return model

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
epochs = 15
lrate = 0.01
decay = lrate/epochs
model = createmodel()
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
batch_size = 32

history=model.fit(X_train, Y_train, epochs =5, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
print(score)
print(acc)
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r--')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
