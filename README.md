# Cerebria: Transforming Medicine
This code is an implementation of a brain tumor classification model using a pre-trained VGG16 model with the TensorFlow/Keras framework. The model is trained to classify brain tumor images into two categories: benign and malignant.

## Prerequisites
Before running the code, make sure you have the following installed:
pandas
numpy
matplotlib
tensorflow
keras

## Description
Import necessary libraries: The code begins by importing the required libraries, including pandas, numpy, os, tensorflow, keras, and matplotlib.

Model Definition: The code then defines the VGG16 model with its top layers removed. It loads the VGG16 model with pre-trained weights from the ImageNet dataset and prepares the model for transfer learning.

Model Architecture: The code adds a Global Average Pooling 2D layer followed by three Dense layers with ReLU activation functions. The final Dense layer has two units with a softmax activation function for binary classification (benign and malignant).

Data Preparation: The code sets up an ImageDataGenerator to preprocess and augment the data. The training and validation data are loaded from the 'data/' directory, with a validation split of 20%.

Model Compilation: The model is compiled using the Adam optimizer with a categorical crossentropy loss function and accuracy as the metric.

Training: The model is then trained using the fit_generator function. The training data is split into batches, and the training and validation steps are calculated based on the batch size and number of samples. The training is performed for five epochs, and a ModelCheckpoint is used to save the best model based on the validation loss.

## Data Preparation
The brain tumor dataset is prepared using the ImageDataGenerator from TensorFlow/Keras. The data is loaded from the 'data/' directory, with a validation split of 20%. The images are preprocessed and augmented to increase the dataset size and improve model generalization.

## Training and Saving the Model
To train the model and save the best model based on validation loss, run the provided code. The model will be saved as 'best_model.h5' in the current directory.

Note: The number of epochs (currently set to 5) can be adjusted based on the dataset size and convergence of the model.

Remember to have a sufficiently large and diverse dataset for training a robust model. For medical applications, it is essential to ensure that the dataset is well-balanced and labeled accurately.

## Results
The brain tumor classification model achieved an impressive validation accuracy of 99.24%. The high accuracy demonstrates the effectiveness of transfer learning with the VGG16 model and the quality of the brain tumor dataset used for training.

Please note that this code is intended for educational purposes and may require further optimization and validation for real-world medical applications. For medical use, it is essential to consult with medical professionals and adhere to ethical guidelines.
