**Handwritten Digit Recognition using Machine Learning**
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using Keras and TensorFlow.

**Project Overview**

1.Load and preprocess the MNIST dataset.
2.Build a CNN model with convolutional, pooling, and dense layers.
3.Train the model to classify digits (0-9).
4.Evaluate and visualize the performance.

**Dataset**                                          

The project uses the MNIST dataset, which consists of:
60,000 training images
10,000 test images
**Image dimensions**: 28x28 pixels, grayscale

**Model Architecture**
**The CNN model consists of:**
Conv2D (64 filters, 3x3 kernel, ReLU activation)
Conv2D (32 filters, 3x3 kernel, ReLU activation)
MaxPooling2D (2x2 pool size)
Flatten layer
Dense layer (10 output units, softmax activation)

**Training**
Optimizer: Adam
Loss function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 10

**Results**
After training, the model achieves around 98% accuracy on the test set
