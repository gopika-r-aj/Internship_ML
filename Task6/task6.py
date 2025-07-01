# Task 6 : KNN using Iris dataset (from local CSV)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load CSV dataset and normalize features
df = pd.read_csv("IrisDataset/Iris.csv") 
X = df.drop("Species", axis=1)  
y = df["Species"]

# Encode string labels to numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_  # ['setosa', 'versicolor', 'virginica']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Create pipeline with StandardScaler and KNN model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Try multiple values of K using GridSearchCV
param_grid = {"knn__n_neighbors": range(1, 21)}
grid = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
grid.fit(X_train, y_train)

best_k = grid.best_params_["knn__n_neighbors"]
best_model = grid.best_estimator_
print(f" Best k: {best_k} (CV accuracy: {grid.best_score_:.3f})")

# Evaluate model on test data
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f" Test Accuracy: {acc:.3f}")
print(" Confusion Matrix:\n", cm)

# Display Confusion Matrix visually
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot decision boundary (only for 2D feature pair)
def plot_decision_boundary(model, X, y, f1=2, f2=3, h=0.02):
    x_min, x_max = X.iloc[:, f1].min() - 1, X.iloc[:, f1].max() + 1
    y_min, y_max = X.iloc[:, f2].min() - 1, X.iloc[:, f2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prepare grid input
    X_plot = np.zeros((len(xx.ravel()), X.shape[1]))
    X_plot[:, f1] = xx.ravel()
    X_plot[:, f2] = yy.ravel()
    X_plot = pd.DataFrame(X_plot, columns=X.columns)

    Z = model.predict(X_plot)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X.iloc[:, f1], X.iloc[:, f2], c=y, edgecolor='k')

    # Correct label mapping
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor=plt.cm.tab10(i), markersize=8)
                       for i, label in enumerate(class_names)]
    plt.legend(handles=legend_elements, title="Classes")

    plt.xlabel(X.columns[f1])
    plt.ylabel(X.columns[f2])
    plt.title(f"KNN Decision Boundary (k={best_k})")
    plt.tight_layout()
    plt.show()


plot_decision_boundary(best_model, X, y_encoded)
