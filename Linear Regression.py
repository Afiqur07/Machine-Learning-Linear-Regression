#
#  Assignment 2
#
#  Group 33:
#  <SM Afiqur Rahman> <smarahman@mun.ca>
#  <Jubaer Ahmed Bhuiyan> <Group Member 2 email>

####################################################################################
# Imports
####################################################################################
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################


def classify():
    print('Performing Regression...')


def Q1_results():
    print('Generating results for Q1...')
    # Load the data from csv file
    data = pd.read_csv('train.csv')

    # Split the data into input (X) and output (y) variables
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split the data into train and test subsets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Split the train subset into train and validation subsets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Train the linear regression model using OLS method
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict the test set results
    y_pred = regressor.predict(X_test)

    # Calculate the residual standard error (RSE) and R2 statistic using the test set
    rse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_val = r2_score(y_test, y_pred)

    # Evaluate model performance using validation approach
    err_val = np.mean((y_pred - y_test)**2)

    # Evaluate model performance using cross-validation approach with 5 folds
    err_cv = np.abs(np.mean(cross_val_score(regressor, X_train_val,
                    y_train_val, scoring='neg_mean_squared_error', cv=5)))
    rse_cv = np.sqrt(err_cv)
    r2_cv = np.abs(np.mean(cross_val_score(
        regressor, X_train_val, y_train_val, scoring='r2', cv=5)))

    # Print the mean squared error, RSE, and R2 statistic for both approaches
    print("Validation approach error (MSE):", err_val)
    print("Validation approach RSE:", rse_val)
    print("Validation approach R2:", r2_val)
    print("Cross-validation approach error (MSE):", err_cv)
    print("Cross-validation approach RSE:", rse_cv)
    print("Cross-validation approach R2:", r2_cv)


def Q2_results():
    print('Generating results for Q2...')
    # Load the training and test datasets
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # Separate the input features and the output variable from the datasets
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Perform CV-based grid search to tune the regularization parameter "α" of the Ridge regression model
    alphas = 10**np.linspace(-5, 5, 100)
    param_grid = {'alpha': alphas}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Extract the best value of "α" from the grid search results
    best_alpha = grid_search.best_params_['alpha']

    # Train a Ridge regression model using the entire training dataset and the best value of "α"
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_train, y_train)

    # Evaluate the performance of the final Ridge regression model on the test dataset
    y_pred = ridge_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("MSE on test data:", mse)
    print("RSE on test data:", rse)
    print("R2 on test data:", r2)

    # Plot the performance of the Ridge regression models explored during the "α" hyperparameter tuning phase as a function of "α"
    cv_results = grid_search.cv_results_
    cv_alphas = cv_results['param_alpha'].data
    cv_scores = np.sqrt(-cv_results['mean_test_score'])

    plt.semilogx(cv_alphas, cv_scores)
    plt.xlabel('alpha')
    plt.ylabel('RMSE')
    plt.title('Ridge regression CV Results')
    plt.show()


def Q3_results():
    print('Generating results for Q3...')
    # Load the training and test datasets
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # Split the training dataset into features (X) and target variable (y)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    # Split the test dataset into features (X) and target variable (y)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Define a range of alpha values to explore
    alphas = np.logspace(-4, 2, 100)

    # Train the Lasso regression model with different alpha values
    rse_list = []
    r2_list = []
    coef_list = []
    for alpha in alphas:
        # set the regularization parameter alpha
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        rse_list.append(rse)
        r2_list.append(r2)
        coef_list.append(lasso.coef_)

    # Plot the performance metrics as a function of alpha
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, rse_list, label='RSE')
    plt.semilogx(alphas, r2_list, label='R2')
    plt.xlabel('Alpha')
    plt.ylabel('Performance Metrics')
    plt.title('Performance of Lasso Regression Models')
    plt.legend()
    plt.show()

    # Choose the best alpha value based on the RSE or R2 plot
    best_alpha_rse = alphas[np.argmin(rse_list)]
    best_alpha_r2 = alphas[np.argmax(r2_list)]
    print('Best alpha based on RSE:', best_alpha_rse)
    print('Best alpha based on R2:', best_alpha_r2)

    # Train the final Lasso regression model using the best alpha value
    best_alpha = best_alpha_rse  # or best_alpha_r2
    # set the regularization parameter alpha
    lasso = Lasso(alpha=best_alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    coef = lasso.coef_

    # Report the performance metrics and model coefficients
    print('Residual standard error:', rse)
    print('R2 score:', r2)
    print('Coefficients:', coef)


def predictCompressiveStrength(Xtest, data_dir):
    # Load the training and test datasets
    train_data = pd.read_csv(data_dir + "/train.sNC.csv", header=None)
    test_data = pd.read_csv(data_dir + "/train.sDAT.csv", header=None)

    # Separate the input features and the output variable from the datasets
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Perform CV-based grid search to tune the regularization parameter "α" of the Ridge regression model
    alphas = 10**np.linspace(-5, 5, 100)
    param_grid = {'alpha': alphas}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Extract the best value of "α" from the grid search results
    best_alpha = grid_search.best_params_['alpha']

    # Train a Ridge regression model using the entire training dataset and the best value of "α"
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_train, y_train)

    # Evaluate the performance of the final Ridge regression model on the test dataset
    y_pred = ridge_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return (y_pred)


#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
