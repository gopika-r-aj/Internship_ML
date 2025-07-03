# Breast Cancer Classification using Support Vector Machines (SVM)

This task demonstrates the use of **Support Vector Machines (SVM)** for binary classification using the **Breast Cancer dataset**.

The goal is to classify tumors as **malignant** or **benign** using both **linear and non-linear (RBF) SVMs**, evaluate their performance, and visualize the decision boundary in 2D using PCA.

## Objectives

This task covers:

1.  Loading and preparing a dataset for binary classification  
2.  Training an SVM with linear and RBF kernel  
3.  Visualizing the decision boundary using 2D PCA  
4.  Tuning hyperparameters like `C` and `gamma`  
5.  Using cross-validation to evaluate model performance

## Dataset

- **Source**: [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- **Target Variable**: `diagnosis` (M = Malignant, B = Benign)

## Tools & Libraries Used

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (`SVC`, `GridSearchCV`, `PCA`, `cross_val_score`)


##  Results
The result will be:
- Print accuracy scores
- Show classification reports
- Display a 2D decision boundary plot.

The screenshots are added to 'Output_screenshots'.