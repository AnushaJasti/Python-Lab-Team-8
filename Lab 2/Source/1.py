from pandas._libs.reduction import apply_frame_axis0
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy import cov
data=pd.read_csv('train.csv')

data['Embarked'] = data['Embarked'].map({'Q': 1, 'S': 0 ,'C':2})
data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}) #replacing the categorical values with numerical
data.select_dtypes(include=[np.number]).interpolate().dropna() #replacing the null values with the average values
print(data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
print(data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
data['Age'].fillna(data['Age'].mean(),inplace =True)
data = data.dropna(how='any',axis=0)
print(data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
data=data.drop(columns=['Name','SibSp','Parch','Ticket','Fare','Cabin']) #droping the attribute which are not required
train,test=train_test_split(data,test_size=0.2)
train_label=train['Survived']
train=train.drop(columns=['Survived'])
test_label=test['Survived']
test=test.drop(columns=['Survived'])
NBclf=GaussianNB()# fitting Naive bayes classifier
NBclf.fit(train,train_label)
SVCclf=SVC() #fitting Support Vector Classifier
SVCclf.fit(train,train_label)
KNNclf=KNeighborsClassifier(n_neighbors=2) #fitting K nearest neighbour classifier
KNNclf.fit(train,train_label)
nbscore= NBclf.score(test,test_label)
svcscore=SVCclf.score(test,test_label)
knnscore=KNNclf.score(test,test_label)
print("Score for Naive Bayes is: ", nbscore)
print("Score for Support Vector Classifier is: ",svcscore)
print("Score for KNN Classifier is : ",knnscore)
