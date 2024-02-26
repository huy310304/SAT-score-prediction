# SAT Score Prediction Project

#### TODO: Clean stuff, save model and create UI, deploy w Docker to actually host, run, save and compare results. 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prequisites](#prequisites)
- [Content](#content)
- [Results Discussion](#results-discussion)
- [Sources](#sources)


## Overview

SAT Score Prediction Project using raw neural networks constructed using Numpy and PyTorch with both regression and classification methods. This project aims to predict SAT scores based on GPA and other factors using a machine learning model built with the PyTorch framework. The model utilizes a neural network architecture to make accurate predictions and has been fine-tuned to achieve high accuracy and fast convergence.


## Features
- **PyTorch Neural Network:** Developed a multi-layer neural network architecture using PyTorch for SAT score prediction.

- **Data Preprocessing:** Implemented comprehensive data preprocessing techniques such as binary splitting and Principal component analysis (PCA) to handle missing values, outliers, and feature scaling.

- **Hyperparameter Tuning and Model Design:** Conducted 100+ hyperparameter fine-tuning and trying different combinations of neural network models, loss function, optimizer, and learning rate scheduler to optimize the model's performance, achieving an accuracy of 90%.

- **Advanced Activation Functions:** Utilized advanced activation functions, such as Leaky ReLU, to enhance model training and convergence.

- **Dropout Regularization:** Applied dropout layers to prevent overfitting and improve the generalization capability of the model.

- **Reduce Learning Rate:** Implemented a learning rate scheduler to gradually reduce the learning rate during training, improving convergence and final performance.

- **Data Randomization and Generation:** Generated synthetic data and introduced randomization techniques to enhance model robustness and generalization, and prevent overfitting.

- **Evaluation:** Using R-squared value and MAE error to evaluate the accuracy of the model over a set of 10% test data.

- **Visualization:** Created informative plotting visualizations of training and validation metrics to analyze the model's performance.

- **User Demo:** Created a CLI demo of predicting the SAT score by taking in user's inputs.  


## Prequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later installed
- Required Python packages listed in `requirements.txt`
- CSV files listed in `data`


## Content
### `data`: contains CSV files for training inputs
- GPA_Small.csv: contains the original CSV file.
- GPA_dummies.csv: generative data used to prevent overfitting.
- GPA_Big.csv: final data file with multiple generative data combined with original data.

### `SAT_score_scratch`: original neural network framework constructed using NumPy library
  
### `classification`: classification models dividing SAT score into 12 brackets of range 100

### `regression`: regression models with multiple data preprocessing techniques

## Results Discussion

In the pursuit of predicting SAT scores based on GPA and other factors, rigorous experimentation and analysis were conducted. All model runs were performed with a consistent training setup of 200 epochs, capturing valuable insights into the convergence, accuracy, and performance over time.

### [1. Initial Model Exploration](./SAT_score_scratch/)

The initial phase of the project involved training a neural network model from scratch using NumPy.

### [2. Regression Models](./regression/)
Optimized **Regression** Neural Network Models Achieving 90% Accuracy with Advanced Techniques and Enhanced Data Preprocessing.

### [3. Classification Models](./classification/)
Optimized **Classification** Neural Network Models with Advanced Techniques and Enhanced Data Preprocessing for comparison.

### Conclusion

The combination of regression and classification approaches, along with rigorous model optimization and preprocessing techniques, showcases the project's progress in predicting SAT scores using regression. The achievements and insights gained underline the potential impact of these models in educational evaluation and assessment, while also highlighting the challenges associated with certain classification tasks.

## Sources
- The framework in `SAT_score_scratch` is adapted from the [BikeSharing Project](https://github.com/udacity/deep-learning-v2-pytorch/tree/c9404fc86181fc3f0906b368697268257f348535/project-bikesharing) by Udacity
