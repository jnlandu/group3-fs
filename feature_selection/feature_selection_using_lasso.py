# -*- coding: utf-8 -*-
"""Feature Selection using Lasso.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LjrI50D8OkISyyVdcv9J-PXz4tvY8QHH

### loading required dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')
# %matplotlib inline

"""# Loading and preparing the dataset"""

house_price = pd.read_csv('house-prices.csv')

house_price.shape

house_price.info()

house_price.head()

house_price.describe()

"""### Lets plot some graph for the EDA purpose"""

plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
plt.scatter(house_price.MasVnrArea,house_price.SalePrice)
plt.subplot(2,3,2)
plt.scatter(house_price.TotalBsmtSF,house_price.SalePrice)
plt.subplot(2,3,3)
plt.scatter(house_price['1stFlrSF'],house_price.SalePrice)
plt.subplot(2,3,4)
plt.scatter(house_price['GarageArea'],house_price.SalePrice)
plt.subplot(2,3,5)
plt.scatter(house_price['GrLivArea'],house_price.SalePrice)
plt.subplot(2,3,6)
plt.scatter(house_price['BsmtFinType2'],house_price.SalePrice)

"""### Plotting heatmap to check the corellation between numerical variables"""

plt.figure(figsize=(32,32))
sns.heatmap(house_price[list(house_price.dtypes[house_price.dtypes!='object'].index)].corr(),annot=True)
plt.show()

"""### Performing one-hot encoding (OHE) on the categorical variables
 Categorical attributes should be transformed into a numerical representation that can be used by the regression models.



"""

house_price[list(house_price.dtypes[house_price.dtypes=='object'].index)].head()

cat_columns = ['MSZoning','LandContour','LotConfig','Neighborhood','Condition1','Condition2','BldgType',
             'HouseStyle','RoofStyle','RoofMatl','Exterior1st',  'Exterior2nd','MasVnrType','Foundation',
             'Heating','Electrical','Functional','GarageType','PavedDrive','SaleType','SaleCondition']

dummy_col = pd.get_dummies(house_price[cat_columns],drop_first=True)
house_price = pd.concat([house_price,dummy_col],axis='columns')
house_price = house_price.drop(cat_columns,axis='columns')

house_price.info()

"""### Creating train and test dataset for validation purpose

Allocating 70% of the data to the training set (df_train), and 30% to the testing set (df_test).
"""

df_train,df_test = train_test_split(house_price,train_size=0.7,test_size=0.3,random_state=42)

"""### Scaling features

#### Let us check the distribution of our target variable (house sale price) before scaling.
"""

plt.figure(figsize=(4,3))
sns.distplot(house_price.SalePrice)
plt.show()

"""#### Applying StandardScaler
A class from scikit-learn that standardizes features by removing the mean and scaling to unit variance.
"""

num_col = ['MSSubClass','LotArea','OverallQual','OverallCond',
           'MasVnrArea','BsmtFinSF1',
           'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
           'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
           'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
           'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
           'ScreenPorch','MiscVal','SalePrice']

scaler = StandardScaler()
df_train[num_col] = scaler.fit_transform(df_train[num_col])
df_test[num_col] = scaler.transform(df_test[num_col])

"""#### Lets check the distribution again after scaling"""

plt.figure(figsize=(10,3))
plt.subplot(121)
sns.distplot(df_train.SalePrice)
plt.subplot(122)
sns.distplot(df_test.SalePrice)

"""#### Spliting the dependent and independent variable"""

y_train = df_train.pop('SalePrice')
y_test = df_test.pop('SalePrice')
X_train, X_test = df_train, df_test

"""## Recursive Feature Elimination (RFE)


RFE is a feature selection method from scikit-learn's feature_selection module. It is considered a wrapper method because it recursively eliminates the least important features until a specified number of features is reached.
"""

from sklearn.feature_selection import RFE

"""These lines create a linear regression model (lm) and fit it to the training data (X_train and y_train). This is necessary because RFE requires a fitted estimator to perform feature selection."""

model  = LinearRegression()
model.fit(X_train,y_train)

"""An RFE object is created with the following parameters:
+ estimator: the fitted linear regression model (lm).
+ n_features_to_select: the number of features to select (in this case, 70).
"""

rfe = RFE(estimator=model, n_features_to_select=70)
rfe.fit(X_train,y_train)

"""#### Examining the results of the RFE scores for our data features and showing whether each feature is selected or not and the ranking of each feature"""

rfe_scores = pd.DataFrame(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))
rfe_scores.columns = ['Column_Names','Status','Rank']
rfe_scores

"""#### The selected features"""

rfe_sel_columns = list(rfe_scores[rfe_scores.Status==True].Column_Names)
len(rfe_sel_columns)

"""#### Filtering the train and test set for the RFE selected columns"""

X_train = X_train[rfe_sel_columns]
X_test = X_test[rfe_sel_columns]

"""# Lasso regression model"""

model = Lasso()

"""#### Hyperparameter tuning for a Lasso regression model using GridSearchCV and K-Fold cross-validation."""

hyper_param = {'alpha':[0.001, 0.01, 0.1,1.0, 5.0, 10.0,20.0]}

folds = KFold(n_splits=10,shuffle=True,random_state=42)
model_cv = GridSearchCV(estimator = model,
                        param_grid=hyper_param,
                        scoring='neg_mean_squared_error',
                        cv=folds,
                        verbose=1,
                        return_train_score=True
                       )

model_cv.fit(X_train,y_train)

""" Results of the grid search, including the hyperparameter combinations, scores, and other metrics."""

cv_result_l = pd.DataFrame(model_cv.cv_results_)
cv_result_l['param_alpha'] = cv_result_l['param_alpha'].astype('float32')
cv_result_l.head()

plt.figure(figsize=(6,4))
plt.plot(cv_result_l['param_alpha'],cv_result_l['mean_train_score'])
plt.plot(cv_result_l['param_alpha'],cv_result_l['mean_test_score'])
plt.xscale('log')
plt.ylabel('R2 Score')
plt.xlabel('Alpha')
plt.show()

# taking the best parameter(Alpha value)
best_alpha = model_cv.best_params_['alpha']
best_alpha

"""#### Fitting the Lasso regression model using the best hyperparameter value found by GridSearchCV"""

model = Lasso(alpha=best_alpha)
model.fit(X_train,y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(r2_score(y_true=y_train,y_pred=y_train_pred))
print(r2_score(y_true=y_test,y_pred=y_test_pred))

"""### Visualize the feature importance and feature selection by Lasso regression model."""

model_parameter = list(model.coef_)
model_parameter.insert(0,model.intercept_)
model_parameter = [round(x,3) for x in model_parameter]
col = df_train.columns
col.insert(0,'Constant')
feat_weights = list(zip(col,model_parameter))

"""Displaying considered features using a threshold of 0.01"""

selected_features = [x for x in feat_weights if abs(x[1]) > 0.01]
selected_features

rejected_features = [x for x in feat_weights if abs(x[1]) <= 0.01]
rejected_features