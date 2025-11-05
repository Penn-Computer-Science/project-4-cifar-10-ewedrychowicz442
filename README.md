[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/1IJxXOgr)
# TiCs25P4
Topics in CS: Project 4: Artificial Intelligence


# CIFAR-10 Classification Assignment

## Overview
In this assignment, you’ll adapt your existing MNIST classification code to work with the **CIFAR-10** dataset — a more complex set of 60,000 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

Your goal is to:
1. Load and preprocess the CIFAR-10 dataset.
2. Modify your MNIST model to handle 32×32×3 color images.
3. Train and evaluate your model.
4. Compare results with your MNIST model.

---

## Step 1: Load the Dataset
Use the built-in Keras dataset loader:
```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

## Normalize the images to [0,1]
```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```
Print the dataset shapes and visualize a few samples using matplotlib


## Step 2. Modify your MNIST Model
  -  Input shape is 32x32x3, not 28 x 28 x 1
  -  Convolutional layers should have 64 filters, not 32
  -  Use **Pooling** and **Dropout** as needed to prevent overfitting
  -  Output layer should still be **Dense** with 10 classes and a **softmax** activation

## Step 3. Compile and Train

  -  Use Adam optimzer and appropriate the loss
```python
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
  -  Train for *at least* 10 epochs
```python
history = model.fit(x_train, y_train,epochs=10,validation_data=(x_test, y_test))
```
  -  Save your training history and visualize
    -  Accuracy over epochs
    -  Loss over epochs
## Step 4. Evaluate and Visualize Results
  -  Print test acuracy
  -  Display a few sample preditions alongisde their true labels
  -  Create at least one plot comparing training and validation accuracy
## Step 5. Reflection
 -  Add a short Markdown section at the end of your notebook:
   -  What differences did you notice between MNIST and CIFAR-10?
   -  What changes to your architecture helped performance?
   -  If you had more time, what improvements would you try next?

## Submission:
Be sure to include the following in your submission - Push to GitHub with a comment of "FINISHED" when complete
  -  cifar10.py -  Your python script
  -  Model architecture summary - use model.summary()
  -  Trianing plots - accuracy and loss vs. epochs
  -  Test accuracy - final evaluation score
  -  Sample predictions -  visual examples of the model output
  -  Reflection section -  written markdown cell with your observations

## Rubric (Total: 100 points)

| Category | Points | Description |
|-----------|:------:|-------------|
| **Data Loading & Preprocessing** | **15** | Properly loads CIFAR-10 data, normalizes inputs, and visualizes a few samples. |
| **Model Architecture** | **25** | CNN architecture adapted from MNIST, with appropriate layers for color images (32×32×3). |
| **Training & Evaluation** | **25** | Model compiles, trains for required epochs, and reports training/validation accuracy. |
| **Visualization** | **15** | Includes at least one accuracy/loss plot and sample prediction visualizations. |
| **Reflection** | **20** | Clear written discussion of differences between MNIST and CIFAR-10, performance observations, and ideas for improvement. |
| **Total** | **100** | — |





