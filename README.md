!pip install Tensorflow

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from  tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
import random as r

x_train = np.loadtxt("input.csv",delimiter=",")
y_train=np.loadtxt("labels.csv",delimiter=",")
x_test=np.loadtxt("input_test.csv",delimiter=",")
y_test = np.loadtxt("labels_test.csv",delimiter=",")

print("x_train.shape:",x_train.shape)
print("y_train.shape:",y_train.shape)
print("x_test.shape:",x_test.shape)
print("y_test.shape:",y_test.shape)
# 3000 is 100*100*3 (RGB)image

Reshape


x_train = x_train.reshape(len(x_train),100,100,3)
y_train = y_train.reshape(len(y_train),1)
x_test = x_test.reshape(len(x_test),100,100,3)
y_test = y_test.reshape(len(y_test),1)
x_train=x_train/255.0
x_test = x_test/255.0

print("x_train.shape:",x_train.shape)
print("y_train.shape:",y_train.shape)
print("x_test.shape:",x_test.shape)
print("y_test.shape:",y_test.shape)
# 3000 is 100*100*3 (RGB)image

x_train

r_i = r.randint(0, len(x_train))
plt.imshow(x_train[r_i,:])
print(plt.show())

 # model

model = Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(100,100,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])         

model.fit(x_train,y_train,epochs=5,batch_size = 64)

model.evaluate(x_test,y_test)

# making prediction 

idx = r.randint(0,len(y_test))
plt.imshow(x_test[idx,:])
plt.show()
y_prd =model.predict(x_test[idx,:].reshape(1,100,100,3))
y_prd=y_prd>0.5
if(y_prd==0):
    pred = "dog"
else:
    pred="cat"
print(pred)
