import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
data= pd.read_csv('Admission_Predict.csv',header=None).values
X_train, X_test, Y_train, Y_test = train_test_split(data[1:,0:8], data[1:,8],
                                                 test_size=0.25, random_state=87)
def createmodel():
    model=Sequential()
    model.add(Dense(8,input_dim=8,init='normal',activation='sigmoid'))
    model.add(Dense(13,init='normal',activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    return model
tensorboard = TensorBoard(log_dir="logslo1/1b",histogram_freq=0, write_graph=True, write_images=True)

estimator=KerasRegressor(build_fn=createmodel)
est=estimator.fit(X_train,Y_train,epochs= 10, batch_size= 130,callbacks=[tensorboard])
evaluation= estimator.score(X_test,Y_test)
print(evaluation)
plt.plot(est.history['loss'])
# plt.plot(history.history['test_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()