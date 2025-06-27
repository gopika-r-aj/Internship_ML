# Logistic Regression Classifier - Breast Cancer Detection

This project implements a **binary classification** model using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset**. It demonstrates how to train, evaluate, and interpret a logistic regression classifier using Scikit-learn and visual tools.

## Objective

Build a logistic regression model to classify breast cancer as **malignant (0)** or **benign (1)** based on medical features, and evaluate it using various performance metrics.

## Dataset

- Loaded directly using `sklearn.datasets.load_breast_cancer()`


## Tools & Libraries

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn


## How It Works

### 1. Data Preprocessing
- Train/test split (80/20)
- Feature standardization using `StandardScaler`

### 2. Model Training
- Logistic Regression model fitted on training data

### 3. Evaluation Metrics
- **Confusion Matrix**
- **Precision** and **Recall**
- **Classification Report**
- **ROC Curve** and **AUC Score**

### 4. Threshold Tuning
- Manually adjusted classification threshold (e.g., `0.6`) to analyze its effect on precision and recall

### 5. Sigmoid Function
- Explained and plotted the **Sigmoid Curve**, used in logistic regression to model probabilities


## Output
The outputs are stored in 'Outputs_screenshots'.
    - Terminal outputs-Confusion matrix (default and custom threshold)
    - ROC Curve
    - Sigmoid Function Plot


