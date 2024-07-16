import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import matplotlib as mlp
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

#load dataset
(x_train,y_train), (x_test,y_test) = mnist.load_data()

#Reshape
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 784)
print(x_train.shape)
x_test = x_test.reshape(x_test.shape[0], 784)
print(x_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#NORMALIZATION
x_train= x_train/255
print(x_train.max())
print(x_train.min())
print(np.unique(x_train))


#Building Model
model = Sequential()
# model.add(Dense(32, input_dim = x_train.shape[1], activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, input_dim = x_train.shape[1],activation= 'sigmoid'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))


#compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

cb = ModelCheckpoint('mymodel.keras', monitor='val_loss', save_best_only=True, mode='min')

# Fit the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[cb])

# Evaluate the model
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#plot
plt.plot(history.history['val_loss'],label='validation_loss')
plt.plot(history.history['loss'],label='train_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()