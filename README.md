# Fashion MNIST Image Classification

# Overview
This project demonstrates an image classification model using the Fashion MNIST dataset. The dataset consists of grayscale images of fashion items, and our goal is to classify these items into ten different categories.

# Requirements
Python 3
Libraries listed in requirements.txt (install them using ```pip install -r requirements.txt``` )
# Dataset
The Fashion MNIST dataset used in this project is available at https://www.kaggle.com/datasets/zalando-research/fashionmnist/data. It includes a training dataset (train.csv) and a test dataset (test.csv).

# Data Analysis
The project starts with data analysis to understand the dataset's structure and characteristics. Exploratory data analysis, insights, and visualizations are presented.

# Model Development
The project includes model development using a Convolutional Neural Network (CNN). The code in model_training.ipynb demonstrates how to train the model using the training data, evaluate its performance on validation data, and save the trained model.

# Model Evaluation
After training, the model's performance is evaluated on the test dataset to assess its ability to generalize to unseen data.

# Active Learning with Expert Feedback
To enhance accuracy and model performance, the project incorporates human expertise through an active learning approach. Uncertain or challenging data points are selected for human review, and expert feedback is integrated into the model's training process.

# Instructions
Install the required Python libraries by running:

```pip install -r requirements.txt```
Load the Fashion MNIST dataset by downloading fashion-mnist_train.csv and fashion-mnist_test.csv from the dataset source and place them in the project directory.
Run the evaluate_model.py file using the following format.

```python evaluate_model.py file_path```
Here file_path must be the folder which contains the dataset. (eg "D:/file") not the path to csv. 

# Author
Faria Sultana
