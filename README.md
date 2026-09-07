# DD2421 Machine Learning — Programming Challenge

A classification project built for the DD2421 Machine Learning course at KTH. The goal is to predict a three-class label from a tabular dataset, using a labeled training set to build and compare several models, then applying the best one to an unlabeled evaluation set.

## Problem
The training data (1000 rows) has 13 features (`x1`–`x13`) and a target `y` with three classes: **Andjorg**, **Andsuto**, and **Jorgsuto**, which are imbalanced (409 / 334 / 257 rows respectively). The features mix continuous numeric values with a nominal categorical variable and a constant column that carries no information.

## Approach
* A correlation check showed `x5` was highly redundant with `x1`, so it was dropped, along with `x12`, which turned out to be constant across every row.
* The categorical feature `x7` was one-hot encoded rather than ordinal encoded, since its five categories have no natural order.
* Before training anything, I compared the distributions of each feature between the training and evaluation sets using density plots, and checked the category frequencies of the one-hot encoded variable too. This was to catch any mismatch between the two sets early, since a model trained on one distribution and evaluated on a visibly different one is a common source of misleading results.
* The target was label encoded, and every model below was evaluated with 5-fold stratified cross-validation rather than a single train-test split, to get a more reliable estimate of how each one generalizes.

## Models Compared

| Model | CV Accuracy | Std Dev |
| :--- | :--- | :--- |
| Decision Tree | 79.0% | 0.023 |
| Neural Network (Keras) | 80.9% | 0.016 |
| Random Forest | 83.9% | 0.010 |
| XGBoost | 84.9% | 0.014 |
| Gaussian Naive Bayes | 85.1% | 0.020 |
| **Mixed Naive Bayes (custom)** | **86.6%** | **0.011** |

## The Winning Model
The best result came from a Naive Bayes variant built specifically for this data instead of relying on scikit-learn's default. Standard Gaussian Naive Bayes assumes every feature is independent given the class, which is rarely true and was likely costing some accuracy here. 

The custom version instead fits a full multivariate Gaussian per class (via a one-component `GaussianMixture`) for the numeric features, which lets it capture how those features move together within each class. The one-hot encoded categorical feature is modeled separately with a Bernoulli Naive Bayes, since that's the distribution that actually fits binary indicator variables. The two sets of log-probabilities are combined at prediction time.

This is the model used to generate the final predictions on the evaluation set, trained on the full 1000-row training set.

