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


print(tf.__version__) #validates that we have tensorflow installed by printing out the version

mnist = tf.keras.datasets.mnist #pulls down the MNIST dataset, it is contained in the tensorflow library
(x_train, y_train), (x_test, y_test) = mnist.load_data() #building a data frame that is seperated into two buckets (training and testing)
#X_train = images in training, y_train = labels of training images
#X_test = images in testing, y_test = labels of testing images

sns.countplot(x = y_train) #x-axis of the chart is the labels of the training images
plt.show()

#check to make sure that there are NO values that are not a number (NaN)

print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing:", np.isnan(x_test).any())

#tell the model what shape to expect
input_shape = (28, 28, 1) #28 pixels wide, 28 pixels tall, 1 color channel (grayscale) - 3 for RGB

#reshaape the training and testing data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) #60,000 images each 28x28 pixels with 1 color channel
x_train = x_train/255.0 #normalize the data to be between 0 and 1
#same for testing data
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) #60,000 images each 28x28 pixels with 1 color channel
x_test = x_test/255.0 #normalize the data to be between 0 and 1

#convert our labels to be one-hot, not sparse
y_train = tf.one_hot(y_train.astype(np.int32), depth = 10) #10 possible labels (0-9), forcing it to be an 32 bit integer
#same for testing labels
y_test = tf.one_hot(y_test.astype(np.int32), depth = 10) #10 possible labels (0-9), forcing it to be an 32 bit integer

#show an example image from MNIST
plt.imshow(x_train[random.randint(0, 59999)][:, :, 0], cmap = 'gray') #showsa random image in the training set, all rows and columns, color channel 0 (grayscale), cmap makes it gray
plt.show()

batch_size = 128 #number of images to process at a time, then upadates weights (more complicated images require smaller batch sizes)
num_classes = 10 #number of possible labels (0-9)
epochs = 5 #number of times to go through the entire dataset during training

#build the model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape), #first convolutional layer, 32 filters, 5x5 kernel, same padding, relu activation function, input shape doesn't change
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape), #same as above but finer combing (3 x 3)
        tf.keras.layers.MaxPool2D(), #reduces the size of the image
        tf.keras.layers.Dropout(0.25), #randomly turns off 25% of the neurons to prevent overfitting, forces it to actually learn instead of memorize
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #another convolutional layer with 64 filters (kernals), finer filter but same sized comb (more teeth)
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #another convolutional layer with 64 filters (kernals), finer filter but same sized comb (more teeth)
        tf.keras.layers.Flatten(), #flattens the 2D matrix into a 1D array for the dense layers
        tf.keras.layers.Dense(num_classes, activation='softmax') #output layer, 10 possible labels, softmax activation to get probabilities for each label (ALWAYS END WITH DENSE LAYER)
    ]
)

#compile + fit the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc']) #RMSprop optimizer, categorical crossentropy loss function (for one-hot labels), forces it into a vector, accuracy metric (highest accuracy)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test)) #fit the model to the training data, validate with testing data

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