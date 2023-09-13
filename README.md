# Handwritten Neo-Tifinagh Character Recognition using Convolutional Neural Networks

This GitHub repository contains code and resources for a character recognition project using Convolutional Neural Networks (CNNs) to recognize handwritten Neo-Tifinagh characters. Neo-Tifinagh is a script used to write the Tamazight language. The project focuses on training a deep learning model to classify handwritten Neo-Tifinagh characters into their respective classes.

## Data

The dataset used for this project is stored in a MATLAB file named `BDR.mat`. It contains two main components:

- `Data_Set2`: Images of handwritten Neo-Tifinagh characters.
- `Data_Label`: Corresponding labels for each character.

  ![Figure_1a](https://github.com/Yamaximum/Handwritten-Neo-Tifinagh-Character-Recognition-using-Convolutional-Neural-Networks/assets/144939420/d53cde1a-88bf-46e5-b0c4-0566ba622339)

## CNN Model

The Convolutional Neural Network (CNN) architecture used for character recognition is inspired by the NYU Deep Learning for Self-Driving Cars course material available [here](https://github.com/Atcold/NYU-DLSP20/blob/master/06-convnet.ipynb). The model architecture consists of the following layers:

- Convolutional Layer 1 with ReLU activation and max-pooling.
- Convolutional Layer 2 with ReLU activation and max-pooling.
- Fully Connected Layer 1 with ReLU activation.
- Fully Connected Layer 2 with LogSoftmax activation for classification.

## Training

The model is trained over several epochs using Stochastic Gradient Descent (SGD) optimization. During training, the following steps are performed:

- Calculation of loss using negative log-likelihood loss (NLL).
- Backpropagation and parameter updates.
- Evaluation of training progress with loss and accuracy metrics.

## Evaluation

The model's performance is evaluated on the test dataset, including the calculation of test loss and accuracy. The results are visualized with loss and accuracy curves.

## Confusion Matrix

A confusion matrix is generated to assess the model's classification performance, with results displayed as a heatmap.

## Repository Structure

- `BDR.mat`: The dataset file.
- `NeoTifinaghCharacterRecognition.ipynb`: file containing the code.
- `README.md`: This README file providing project details and instructions.

## Usage

You can clone this repository and run the Jupyter Notebook `NeoTifinaghCharacterRecognition.ipynb` to train and evaluate the CNN model for handwritten Neo-Tifinagh character recognition. 

Please note that this project may require additional dependencies such as PyTorch, NumPy, Matplotlib, and Seaborn.

Feel free to explore and use this code for your handwritten Neo-Tifinagh character recognition tasks!
