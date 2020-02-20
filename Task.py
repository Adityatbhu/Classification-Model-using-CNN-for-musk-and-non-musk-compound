import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas_profiling as pp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score

data = pd.read_csv('C:\\Users\\Aditya Tiwari\\Downloads\\musk_csv.csv')

data.head()

data.isna().sum()

# checking correlation using correlation matrix
corr_matrix = data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Finding feature columns index with correlation greater than 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
#drop_val = [column for column in corr.abs().columns if any(corr.abs()[column] > 0.92)]
df = data.drop(columns = drop_val)

df.shape

train,test = train_test_split(df, random_state=85, test_size = 0.2)
Mtrain = train.iloc[:,6596:-1]
Ntrain = train.iloc[:,-1:]
Mtest = test.iloc[:,6596:-1]
Ntest = test.iloc[:,-1:]
Mtrain.shape

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

a=[1]*Mtrain.shape[0]
Mtrain["demo"]=a
Mtrain.shape

b=[1]*Mtest.shape[0]
Mtest["demo"]=b
Mtest.shape

x_train=Mtrain.values.reshape(Mtrain.shape[0],19,6,1)
x_test=Mtest.values.reshape(Mtest.shape[0],19,6,1)

x_train.shape
x_test.shape

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(19,6,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
history = model.fit(x_train,Ntrain,batch_size=32,epochs=8,validation_data=(x_test,Ntest))
score=model.evaluate(x_test,Ntest,verbose=0)
print(score)

#matplotlib Library

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy plot')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

print("f1_score:",f1_score(Ntest,model.predict_classes(x_test),))
print("recall:",recall_score(Ntest,model.predict_classes(x_test),))
print("Validation Loss:",score[0])
print("Validation Accuracy:",score[1])

model.save('/content/drive/My Drive/BDA/Datasets/Tasks/credixco/musk_model.h5')