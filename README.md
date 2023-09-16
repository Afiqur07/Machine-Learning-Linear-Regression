# Machine-Learning / Data Science -Linear-Regression Project
Predicting the performance of various high performance concretes using different Linear regression Models.

The Problem: The goal of this assignment is to explore the development of linear regression models for concrete compressive strength prediction based on the relative amounts of ingredients used in a given concrete mixture and the age of the concrete.

Methods Used: Ordinary least squares (“simple”) linear regression, Ridge Regression and Lasso Regression.

The Data: The data consists of a total of 8 features, among which 7 features relate to the relative amounts
of the ingredients (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate
respectively) in a concrete mixture, while a single feature denotes the age of the concrete. Experimentally determined
compressive strength for the given concrete mixture is provided as the outcome variable. The “training” and “test”
datasets consisting of 800 and 100 samples respectively are given in the train.csv and test.csv files respectively.

A snippet of one of the data files: 

![image](https://github.com/Afiqur07/Machine-Learning-Linear-Regression/assets/27920239/2134dc02-c887-4a08-880a-8bb1f7f521e4)

Question 1 :-
Train a multivariate ordinary least squares (“simple”) linear regression model to predict the compressive strength of
an input concrete mixture based on the relevant features. Using only the “training” dataset, estimate the “Err” using
both the validation approach (i.e., split the “training” dataset into “train/validation/test” subsets, and then train on
“train + validation” subset and test on the “test” subset) as well as using a cross-validation (CV) approach. Discuss the
choice of the number folds used in your CV approach, and compare the “Err” estimates obtained using the validation
and CV approaches.

![image](https://github.com/Afiqur07/Machine-Learning-Linear-Regression/assets/27920239/f316d54e-ab0e-4d5f-947d-23f61cb6600c)

![image](https://github.com/Afiqur07/Machine-Learning-Linear-Regression/assets/27920239/e1f54f51-3429-495e-9815-31ecc66f6f3e)

![image](https://github.com/Afiqur07/Machine-Learning-Linear-Regression/assets/27920239/87fd1db3-fa46-4a5b-b08e-e911d671a2ee)

![image](https://github.com/Afiqur07/Machine-Learning-Linear-Regression/assets/27920239/e4661456-38f7-452e-bf49-3dc2c3bca49e)

Question 2
Now train a multivariate Ridge regression model for the above concrete compressive strength prediction task. Use
the “training” dataset and a CV based grid-search approach to tune the regularization parameter “α” of the Ridge
regression model. Using the “best” “α” setting, re-train on the entire “training” dataset to obtain the final Ridge
regression model. Estimate the “Err” of this final model on the “test” dataset. Plot the performance of the models
explored during the “α” hyperparameter tuning phase as function of “α”. Compare the performance of the Ridge
regression model with that of the “simple” linear regression model.

Answer to Question 2:

![image](https://github.com/Afiqur07/Machine-Learning-Linear-Regression/assets/27920239/cbe31451-4a42-468f-96d3-d75f226dea60)

![image](https://github.com/Afiqur07/Machine-Learning-Linear-Regression/assets/27920239/5c45797c-4bcd-4337-87eb-63a08cdd2890)


Question 3
Repeat the above experiment with a multivariate Lasso regression model. Plot the performance of the models explored
during the “α” hyperparameter tuning phase as function of “α”. Compare the performance of the final Lasso regression
model with that of both the Ridge regression and the “simple” linear regression models.

Answer to Question 3:

![image](https://github.com/Afiqur07/Machine-Learning-Linear-Regression/assets/27920239/c8134e5a-1301-44a7-8de5-a5bb95b27607)


Here are a few findings:
• The lowest R2 score, greatest MSE and RSE values, and worst performance of the three
models all point to the Simple Linear Regression model's performance.
• The reduced MSE and RSE numbers and higher R2 score on the test data show that the
Ridge Regression model performs better than the Simple Linear Regression model. This
indicates that when compared to the Simple Linear Regression model, the Ridge
Regression model is more generalizable to novel data.
• With nearly equal RSE and R2 numbers, the Lasso Regression model performs similarly
to the Ridge Regression model. The Ridge Regression model is marginally better at
predicting the outcome variable than the Lasso Regression model, which has a slightly
smaller RSE number.
The Ridge and Lasso Regression models appear to perform better than the Simple Linear
Regression model overall, with the Lasso Regression model slightly outperforming the Ridge
Regression model in terms of predictive performance.
