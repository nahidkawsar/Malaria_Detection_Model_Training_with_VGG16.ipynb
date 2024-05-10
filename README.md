# Malaria Detection Model Training with VGG16

![Front page](https://github.com/nahidkawsar/Malaria_Detection_Model_Training_with_VGG16.ipynb/assets/149723828/61e5cc86-a6bb-483c-88a0-9d30a2b6622b)

## Overview
This project involves training a deep learning model for malaria detection using the VGG16 architecture. The trained model can classify whether a given cell image is infected with malaria or not. The training process includes preprocessing the dataset, building and training the model, and evaluating its performance.

## Dataset:
The dataset used for training the model is the "Cell Images for Detecting Malaria" dataset from Kaggle. The dataset contains cell images with labels indicating whether the cell is infected with malaria or not. The dataset was downloaded using the Kaggle API and extracted for further processing.

## Kaggle Dataset Information
- Dataset Name: Cell Images for Detecting Malaria
- Dataset Link: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

## Model Architecture
The model architecture used for training is based on the VGG16 convolutional neural network (CNN) architecture. The VGG16 model is pretrained on the ImageNet dataset and fine-tuned for malaria detection. The model consists of convolutional layers followed by fully connected layers and a sigmoid activation function for binary classification.

## Training Process
- The training process involves the following steps:

Data Preparation: The dataset is split into training and validation sets. Image preprocessing is applied to normalize pixel values and resize images to the required input size (150x150).
- Model Building: The VGG16 architecture is instantiated without the fully connected layers. Additional dense layers are added to the model for classification.
- Model Compilation: The model is compiled with the Adam optimizer and binary crossentropy loss function.
Model Training: The model is trained using the training dataset with 10 epochs. Training progress is monitored using accuracy and loss metrics.
- Model Evaluation: The trained model is evaluated on the validation dataset to assess its performance.

## Saving the Model
Once the training is complete, the trained model is saved to a file (malaria_detection_model.h5) for future use. The saved model file contains the model architecture, weights, and optimizer state.

## Usage
To use the trained model for malaria detection:

- Load the saved model using the load_model() function provided by TensorFlow/Keras.
- Preprocess the input image (resize, normalize pixel values).
- Make predictions using the loaded model on the preprocessed image.
## Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV

## predictions:
![2](https://github.com/nahidkawsar/Malaria_Detection_Model_Training_with_VGG16.ipynb/assets/149723828/cd353a26-4b4f-4582-91d3-f60bb54f7888)

### Author
[H.M Nahid kawsar]
