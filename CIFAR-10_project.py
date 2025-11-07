import seaborn as sns 
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("GPU is not available.")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import random 
from keras.utils import to_categorical


print(tf.__version__) #validates that we have tensorflow installed by printing out the version

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#sns.countplot(x = y_train) #x-axis of the chart is the labels of the training images
#plt.show()

#check to make sure that there are NO values that are not a number (NaN)

print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing:", np.isnan(x_test).any())

#tell the model what shape to expect
input_shape = (32, 32, 3) #32 pixels wide, 32 pixels tall, 3 color channel (RGB)

#reshaape the training and testing data
#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) #60,000 images each 32x32 pixels with 1 color channel
x_train = x_train.astype('float32') / 255.0 #normalize the data to be between 0 and 1
#same for testing data
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) #60,000 images each 32x32 pixels with 1 color channel
x_test = x_test.astype('float32') / 255.0 #normalize the data to be between 0 and 1

#cinvert labels to one hot using to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#show an example image from MNIST
plt.imshow(x_train[random.randint(0, 59999)][:, :, 0], cmap = 'gray') #showsa random image in the training set, all rows and columns, color channel 0 (grayscale), cmap makes it gray
plt.show()

batch_size = 80 #number of images to process at a time, then upadates weights (more complicated images require smaller batch sizes)
num_classes = 10 #number of possible labels (0-9)
epochs = 10 #number of times to go through the entire dataset during training

#build the model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #first convolutional layer, 32 filters, 5x5 kernel, same padding, relu activation function, input shape doesn't change
        #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #same as above but finer combing (3 x 3)
        tf.keras.layers.MaxPool2D(), #reduces the size of the image
        tf.keras.layers.Dropout(0.25), #randomly turns off 25% of the neurons to prevent overfitting, forces it to actually learn instead of memorize
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #another convolutional layer with 64 filters (kernals), finer filter but same sized comb (more teeth)
        tf.keras.layers.MaxPool2D(), #reduces the size of the image
        tf.keras.layers.Dropout(0.25), #randomly turns off 25% of the neurons to prevent overfitting, forces it to actually learn instead of memorizetf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #another convolutional layer with 64 filters (kernals), finer filter but same sized comb (more teeth)
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #another convolutional layer with 64 filters (kernals), finer filter but same sized comb (more teeth)
        tf.keras.layers.MaxPool2D(), #reduces the size of the image
        tf.keras.layers.Dropout(0.25), #randomly turns off 25% of the neurons to prevent overfitting, forces it to actually learn instead of memorize
        tf.keras.layers.Flatten(), #flattens the 2D matrix into a 1D array for the dense layers
        tf.keras.layers.Dense(num_classes, activation='softmax') #output layer, 10 possible labels, softmax activation to get probabilities for each label (ALWAYS END WITH DENSE LAYER)
    ]
)

#compile + fit the model
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc']) #adam optimizer is popular, sparse categorical crossentropy is used for multi-class classification
history = model.fit(x_train, y_train,epochs=epochs,validation_data=(x_test, y_test)) #fit the model to the training data, validate on testing data

#plot out training and validation accuracy/loss
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training Loss") #plot training loss in blue
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss") #plot validation loss in red
legend = ax[0].legend(loc='best', shadow=True) #add legend
ax[0].set_title("Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy") #plot training accuracy in blue
ax[1].plot(history.history['val_acc'], color='r', label="Validation Accuracy") #plot validation accuracy in red
legend = ax[1].legend(loc='best', shadow=True) #add legend
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()