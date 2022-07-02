from keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers

# Images are encoded as Numpy Array
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Number of Images and Shapes
print(train_images.shape) # (60000, 28, 28)
print(test_images.shape) # (10000, 28, 28)

# Network 
model = models.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
])

# Compilation
model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
)

# Preparing Image Data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Fit
model.fit(train_images, train_labels, epochs=5, batch_size=128)
# Epoch 1/5
# 469/469 [==============================] - 5s 7ms/step - loss: 0.2562 - accuracy: 0.9268
# Epoch 2/5
# 469/469 [==============================] - 4s 7ms/step - loss: 0.1032 - accuracy: 0.9691
# Epoch 3/5
# 469/469 [==============================] - 4s 8ms/step - loss: 0.0676 - accuracy: 0.9798
# Epoch 4/5
# 469/469 [==============================] - 4s 8ms/step - loss: 0.0494 - accuracy: 0.9850
# Epoch 5/5
# 469/469 [==============================] - 4s 8ms/step - loss: 0.0374 - accuracy: 0.9886
# <keras.callbacks.History at 0x22701253af0>


# Make Some Predictions
# All possible Probabilities of Prediction
# High will win the case
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(predictions[0])
print(predictions[0].argmax())
print(test_labels[0]) # Check
# [1.1706976e-09 3.5098917e-11 2.4403894e-06 5.2516549e-05 5.6744659e-13
#  2.5932634e-08 1.8395179e-14 9.9994493e-01 7.5849531e-09 8.5818435e-08]
# 7
# 7

# Evaluate Test Score
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0686 - accuracy: 0.9790
# test_acc: 0.9789999723434448