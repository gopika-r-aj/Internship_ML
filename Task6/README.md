# KNN Classification on Iris Dataset

In this task, I have implemented the **K-Nearest Neighbors (KNN)** algorithm using the **Iris dataset** (downloaded from Kaggle). The goal was to understand and apply KNN for solving classification problems, tune the model, evaluate its performance, and visualize the results.


## What I Did

- Loaded the **Iris dataset** from a local CSV file
- Preprocessed the data using **feature normalization**
- Used **KNeighborsClassifier** from `scikit-learn`
- Tuned the best value of **K** using **GridSearchCV**
- Evaluated the model with **accuracy** and a **confusion matrix**
- Plotted the **decision boundaries** for better understanding


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib


## Output

- Automatically selected the best `k` value based on cross-validation
- Achieved high accuracy on test data (typically near 100%)
- Plotted:
  - Confusion Matrix
  - Decision Boundary for Petal features
       The screenshots of the output are added to 'Output_screenshots'.