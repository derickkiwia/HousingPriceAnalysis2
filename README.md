# Housing Price Analysis 2
## Project Overview
This project focuses on analyzing housing property features and price data to uncover patterns and potential predictors of housing prices. Using R for data analysis and visualization, the task of this analysis is to develop machine learning models that can accurately predict housing prices based on property features.

## Installation
To run the analysis, you will need to install certain R packages (ggplot2, dplyr, scales, leaps, ISLR, randomForest, tree, AUC, e1071, corrplot, lmtest, glmnet)

## Dataset
The dataset for US housing prices was obtained from Kaggle. It contains house prices for 545 houses from several states in the US and 12 house-specific features. They include 'price' (indicating the sale price of the house), 'area' (total area of the property), 'bedrooms' (number of bedrooms), 'bathrooms' (number of bathrooms), 'stories' (number of floors in the house), and ‘parking' (number of parking spaces). Additionally, it includes nominal binary variables represented as 'yes' or 'no' for features such as 'mainroad' (whether the house is near a main road), 'guestroom' (whether there is a guest room), 'basement' (presence of a basement), 'hotwaterheating' (availability of hot water heating), and 'airconditioning' (presence of air conditioning). All data variables were transformed into numerical data to allow for inclusion in the linear regression model. The binary nominal variables represented as 'yes' or 'no' were encoded with ‘1’ and ‘0’ respectively. The 'furnishingstatus' variable was encoded with ‘3’ for furnished, ‘2’ for semi-furnished, and ‘1’ for unfurnished.

## Analysis 
The dataset for this analysis was split between train and test data following a 70/30 split to test the generated model’s ability to generalise on unseen data. The training data was used for training the Multivariate Linear Regression model (MLR), a Ridge Regression model, a Random Forest model (RF) and a Radial Support Vector Machine model (Radial SVM).

## Results
The Random Forest (RF) model outperforms the other models (MAE, MSE, ME, RMSE), demonstrating superior accuracy and reliability due to its effectiveness in handling non-linear patterns within the data. Although the data shows linear tendencies, non-linear relationships significantly affect the target variable. The RF and Radial SVM models excel at managing these complexities. In contrast, the best performing Ridge Regression model ranks second in error rates, indicating that merely including all variables without addressing non-linear dynamics leads to less accurate predictions. Error analysis shows that Radial SVM and Ridge Regression tend to underestimate house prices, whereas RF provides a more balanced error distribution across different price levels, making it the most reliable model for predicting house prices. Thus, RF is the preferred choice for capturing complex non-linear relationships in this context.

