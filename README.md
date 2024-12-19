# CAR-PRICE-Machine-Learning-project
Car Price Machine Learning project
This project involves building a regression model to predict car prices based on various features such as car specifications, engine details, and other attributes. Multiple machine learning models were tested and evaluated to determine the best-performing model for predicting car prices.

Table of Contents:

Project Overview
Dataset
Modeling Approach
Model Evaluation
Feature Importance Analysis
Hyperparameter Tuning
Installation
Usage
Results


Project Overview:

The goal of this project is to predict the price of cars using machine learning techniques. The dataset contains various features related to car specifications, such as engine size, horsepower, weight, and more. Several regression models are used to predict car prices, and their performance is compared using metrics such as R-squared, MSE (Mean Squared Error), and MAE (Mean Absolute Error).

Dataset:

The dataset used in this project contains various features of cars, including:

Engine size
Horsepower
Weight
Mileage (city and highway)
Car dimensions
Fuel type
And many more
The dataset is pre-processed, including handling missing values, encoding categorical variables, and feature scaling.

Modeling Approach

Data Preprocessing:

Data cleaning (handling missing values)
Feature scaling (using StandardScaler)
Encoding categorical variables (using one-hot encoding)
Skewness transformation (for normalizing features with high skewness)

Modeling:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Support Vector Regressor (SVR)
Hyperparameter Tuning:

Randomized search and grid search were used to tune the hyperparameters of the models to improve their performance.
Model Evaluation:

Evaluated models using R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).


Best Performing Model : 

The Random Forest Regressor performed the best with the highest R² of 0.94 and the lowest MSE of 3.76M. It was also the most stable across different evaluation metrics.

Feature Importance Analysis

The following features were found to be the most significant in predicting car prices:

curbweight
enginesize
horsepower
highwaympg


Hyperparameter Tuning : 

Through hyperparameter tuning, the performance of the Random Forest Regressor and Gradient Boosting Regressor models was improved slightly, with reductions in MSE and increases in R².

Best parameters for Random Forest Regressor:

n_estimators: 100
max_depth: None
min_samples_split: 2
min_samples_leaf: 1
bootstrap: True

Installation :

To run this project, you'll need to have Python 3.x installed along with the following libraries:

pandas
numpy
scikit-learn
matplotlib
seaborn
To install the required libraries, you can use pip:

bash
Copy code
pip install -r requirements.txt
Usage
After cloning the repository, you can run the Python scripts to load the dataset, preprocess it, build and evaluate models. For example:

bash
Copy code
python car_price_prediction.py
This will execute the full pipeline, from loading the dataset to evaluating the models.

Results
The Random Forest Regressor model is recommended for deployment as it offers the best performance in terms of error metrics and predictive accuracy.
