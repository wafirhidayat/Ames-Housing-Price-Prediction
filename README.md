# Project 2: Ames Housing Data and Kaggle Challenge

## Project Statement

This project aims to examine see what factors contribute in predicting property prices in the housing market using the Ames Housing Data.

## Summary of Project
The Ames Housing Dataset is an exceptionally detailed and robust dataset with a total number of 80 different features relating to houses and over 2000 observations. There are many factors involved in real estate pricing. In reality, it is often hard for us to tell which factors are more important and which factors are not. For this project, we will attempt to build a prediction model to predict house prices of Ames, Iowa with supervised predictive modeling techniques. The dataset is from a [Kaggle competition](https://www.kaggle.com/competitions/dsi-us-11-project-2-regression-challenge/overview).

We will construct multiple regression model that will take in several independent variables, predict the sale price of a house. We will then train the models based on the training dataset and validate the model through the validation dataset. Finally, our goal is to best predict the sale prices of the houses in the test set, and our predictions will then be evaluated on Kaggle. From there, we will find out which factors contribute the most in predicting property prices.

------------------------------------------------
## Data

### CSV Datasets

CSV datasets provided for this project:
* [train.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/train.csv): training dataset
* [test.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/test.csv): test dataset 

Processed datasets:
* [training_model.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/training_model.csv): Cleaned and processed training dataset after train/test split
* [validation_model.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/validation_model.csv): Cleaned and processed validation dataset after train/test split 
* [test_kaggle.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/test_kaggle.csv): Cleaned and processed test dataset for prediction 

Kaggle-submitted CSV:
* [submission_linear_reg_01.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/submission_linear_reg_01.csv): 1st submission based on Linear Regression
* [submission_ridge_01.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/submission_ridge_01.csv): 1st submission based on Ridge Regression
* [submission_lasso_01.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/submission_lasso_01.csv): 1st submission based on Lasso Regression
* [submission_lasso_02.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/submission_lasso_02.csv): 2nd submission based on Lasso Regression after RFE
* [submission_lasso_03.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/submission_lasso_03.csv): Final submission based on Lasso Regression after hypertuning parameters
* [submission_enet_01.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/datasets/submission_enet_01.csv): 1st submission based on ElasticNet Regression

### Data Dictionary
 [Detailed Data Dictionary](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)

------------------------------------------------
## Executive Summary
The project is split up into two code notebooks, with one for [Cleaning and Preprocessing](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/code/01_Cleaning_Preprocessing.ipynb) our training and test datasets while the other for [Modelling and Evaluation](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_2/code/02_Modelling_and_Evaluation.ipynb). The project overall segments are as stated below:

### 1. Data Cleaning

The raw data is processed into usable data through null imputation and feature classification for easier data analysis and for preliminary elimination of features.

### 2. Exploratory Analysis 
After cleaning & combining our datasets, we carry out an initial exploratory analysis to identify some information markers from the data by various plots and graphs to visualize the features better. Here, we identify features that could potentially be 'noise' to our model and reduce the possibility in having multi-collinearity. Boxplots & barplots are used for categorical features while scatterplots are used for continuous features. Heatmaps are also used to study correlations between features to decide if interaction terms will be created. 
With each analysis and data visualization, features were dropped that were deemed to be not useful to our model.

### 3. Preprocessing
The dataset goes through the final steps of pre-processing so that it can be put into a model, including one-hot encoding for nominal features, ordinal encoding for ordinal features and scaling of features for continuous features.

### 4. Modelling
The models are evaluated using various metrics namely Cross Validation Score, R2 Score and Root Mean Squared Error (RMSE). The clean and processed train dataset was then put through four regression models; Linear, Ridge, Lasso and ElasticNet. The final production model is then found by comparing and evaluating the model performance whilst also the considering the interpretability of the model. This chosen model is then submitted to Kaggle to get our initial test score.

### 5. Model Improvement
Recursive Feature Evaluation[(RFE)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) was used to fit our chosen model and to essentially pick the features that have the most significance to our model. In this case, the top 50 features were selected and was shown to give an improvement in scores as compared to our standalone model. Following which, hyperparameters were tuned by using GridSearch to fine the best alpha for our model.

### 6. Conclusions
Interpretations of the model coefficients and recommendations are provided to key stakeholders involved in this project.

------------------------------------------------
## Conclusion and Recommendations
The final production model used was the Lasso Regression Model with 50 features selected using Recursive Feature Elimination (Lasso Regression). Even though it showed good metrics scoring result after validating with our valid dataset, the final Kaggle score showed no improvement, especially after the hyperparameter tuning process. This is likely due to the fact that CV models were already used initially to find the best alpha to fit our respective model. Thus, our initial model showed the best result, albeit only a slight change in scores after RFE filtering. Our final best Kaggle score was 23533.

From the coefficients, we see that generally neighbourhoods and area/size of property contributes the most in sale price. Generally, locations that are 'prime' will lead to an increase in sale price and same goes to the size of the property as it shows a positive . Conversely, the features that had the greatest negative effect on sale price were the age of the house and certain roof styles (Mansard). The older the age of property, the lower the sale price and the more type of 'less-quality' roof styles (that are least popular among homeowners), the lower the sale price as well. 

Therefore, we can answer our project statement whereby the features listed in the coefficient dataframe has highlighted several similarly-described features has an impact on saleprice of properties in Ames, Iowa; namely **neighbourhood/location, size & area of properties and building types** contributes the most to the housing market.
Although our model 

Our model is able to predict housing prices relatively well. It can also be fitted based on the needs of the stakeholders. Our dataset does not have some features such as economic indicators, like  employment or wage growth of the area, and interest rates. Based on our outside research, more macro features has an impact on the housing price in the market. An increase in interest rate can influence one's ability to afford a home as the individual's budget will be more focused on managing to pay off additional interest rates (for example, for credit card or short-term loan). [(source)](https://www.opendoor.com/w/blog/factors-that-influence-home-value)

Therefore, we can include such features to better predict housing market based on more macro features, if the needs of our stakeholders require it to. In this case, our stakeholders are catered more to homeowners that may want to find ways to help alleviate their position in selling their properties at a higher price as the features describe more on the description of the property itself. If our stakeholders are more of real estate developers or investors, whose sole purpose is to see intrinsic value due to long-term price appreciation [(source)](https://www.investopedia.com/articles/investing/110614/most-important-factors-investing-real-estate.asp#:~:text=Expected%20cash%20flow%20from%20rental,to%20get%20a%20better%20price), then a more macro view and data is required in which our model is able to run prediction. 

Even with added features, our model will still be able to instantiate them and evaluate if the features are deemed significant in its predictive power. Especially with Lasso Regression Model, it can sieve out noise better by zeroing the coefficient. With proper datasets and training, we can recommend that the model be trained in more macro-related features to boost its predictive power in the housing market.
