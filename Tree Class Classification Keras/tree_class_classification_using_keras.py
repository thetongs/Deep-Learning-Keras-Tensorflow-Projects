# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

# Load Dataset
X = pd.read_csv('tree_class_feats.csv')
Y = pd.read_csv('tree_class_target.csv')
print("Number of records : {}".format(len(X)))
print("Number of features : {}".format(X.shape[1]))


# Network
model = Sequential()
model.add(Dense(10, 
                activation = 'tanh',
                input_dim = 10))
model.add(Dense(5, 
                activation = 'tanh'))
model.add(Dense(1, 
                activation = 'sigmoid'))


model.compile(optimizer='sgd', loss='binary_crossentropy')
model.summary()


history = model.fit(X, Y, 
        epochs = 100,          # use 100 in actual model 
        batch_size = 5, 
        verbose = 1,
        validation_split = 0.2, 
        shuffle = False)
history.history


# Plotting training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Model loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.close()