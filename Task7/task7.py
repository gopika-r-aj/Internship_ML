# SVM using Breast Cancer Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("BreastCancerDataset/breast-cancer.csv") 
df.drop(columns=[col for col in ['id'] if col in df.columns], inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM with Linear Kernel
linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)
print("Linear Kernel Accuracy:", accuracy_score(y_test, linear_pred))

# SVM with RBF Kernel
rbf_svm = SVC(kernel='rbf', C=1, gamma='scale')
rbf_svm.fit(X_train, y_train)
rbf_pred = rbf_svm.predict(X_test)
print("RBF Kernel Accuracy:", accuracy_score(y_test, rbf_pred))

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# Cross-validation
best_svm = grid.best_estimator_
cv_scores = cross_val_score(best_svm, X_scaled, y, cv=5)
print("Cross-Validation Accuracy:", cv_scores.mean())

# PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

svm_2d = SVC(kernel='linear', C=1)
svm_2d.fit(X_pca, y)

# Decision boundary plot
def plot_boundary(model, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm", s=50)
    plt.title(title)
    plt.show()

plot_boundary(svm_2d, X_pca, y, "SVM Decision Boundary (Linear Kernel, PCA)")

# Print full report
print("\nClassification Report (Linear SVM):\n", classification_report(y_test, linear_pred))
print("\nClassification Report (RBF SVM):\n", classification_report(y_test, rbf_pred))

