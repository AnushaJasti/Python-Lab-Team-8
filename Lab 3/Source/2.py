#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import tensorflow as tf

#setting batch and epochs
from sklearn.model_selection import train_test_split

batch_size = 50
nb_classes = 10
nb_epoch = 10

#loading the dataset(mnist)
dataset = pd.read_csv("heart.csv").values

import numpy as np
dataset[1:,:]=np.asarray(dataset[1:,:],dtype=np.float32)
dataset[1:,0]=(dataset[1:,0]-min(dataset[1:,0]))/(max(dataset[1:,0])-min(dataset[1:,0]))
dataset[1:,1]=(dataset[1:,1]-min(dataset[1:,1]))/(max(dataset[1:,1])-min(dataset[1:,1]))
dataset[1:,2]=(dataset[1:,2]-min(dataset[1:,2]))/(max(dataset[1:,2])-min(dataset[1:,2]))
dataset[1:,3]=(dataset[1:,3]-min(dataset[1:,3]))/(max(dataset[1:,3])-min(dataset[1:,3]))
dataset[1:,4]=(dataset[1:,4]-min(dataset[1:,4]))/(max(dataset[1:,4])-min(dataset[1:,4]))
dataset[1:,5]=(dataset[1:,5]-min(dataset[1:,5]))/(max(dataset[1:,5])-min(dataset[1:,5]))
dataset[1:,6]=(dataset[1:,6]-min(dataset[1:,6]))/(max(dataset[1:,6])-min(dataset[1:,6]))
dataset[1:,7]=(dataset[1:,7]-min(dataset[1:,7]))/(max(dataset[1:,7])-min(dataset[1:,7]))
dataset[1:,8]=(dataset[1:,8]-min(dataset[1:,8]))/(max(dataset[1:,8])-min(dataset[1:,8]))
dataset[1:,9]=(dataset[1:,9]-min(dataset[1:,9]))/(max(dataset[1:,9])-min(dataset[1:,9]))
dataset[1:,10]=(dataset[1:,10]-min(dataset[1:,10]))/(max(dataset[1:,10])-min(dataset[1:,10]))
dataset[1:,11]=(dataset[1:,11]-min(dataset[1:,11]))/(max(dataset[1:,11])-min(dataset[1:,11]))
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:13], dataset[:,13],
                                                    test_size=0.25, random_state=87)
Y_Train = np_utils.to_categorical(Y_train, nb_classes)
Y_Test = np_utils.to_categorical(Y_test, nb_classes)
model = Sequential()
model.add(Dense(output_dim=10, input_shape=(13,), init='normal', activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#tensorboard graph genertion
tensorboard = TensorBoard(log_dir="logslo1/2",histogram_freq=0, write_graph=True, write_images=True)
history=model.fit(X_train, Y_Train, nb_epoch=nb_epoch, batch_size=batch_size,callbacks=[tensorboard])

#predicting the accuracy of the model
score = model.evaluate(X_test, Y_Test, verbose=1)
print('Loss: %.2f, Accuracy: %.2f' % (score[0], score[1]))

#plotting the loss
plt.plot(history.history['loss'])
# plt.plot(history.history['test_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()