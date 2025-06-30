#  Decision Trees and Random Forests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from loguru import logger

# Set visual style
sns.set(style="whitegrid")

# Load & Explore Data
def load_data(path='HeartDiseaseDataset/heart.csv'):
    """Loads dataset from given CSV path."""
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

def explore_data(df):
    """Displays basic EDA (nulls, stats, correlations)."""
    logger.info("Exploring dataset...")
    logger.debug(df.describe())
    logger.debug(df.isnull().sum())

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

# Split Data
def split_data(df, target='target', test_size=0.2):
    """Splits dataset into train and test sets."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Decision Tree
def train_decision_tree(X_train, y_train, max_depth=None):
    """Trains a Decision Tree model."""
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    logger.success(f"Trained Decision Tree (max_depth={max_depth})")
    return clf

def visualize_tree(clf, feature_names, class_names):
    """Visualizes the trained Decision Tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

# Random Forest
def train_random_forest(X_train, y_train, n_estimators=100):
    """Trains a Random Forest model."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    logger.success("Trained Random Forest")
    return rf

# Evaluation
def evaluate_model(model, X_test, y_test, title="Model"):
    """Evaluates model with accuracy, classification report, and confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"{title} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    disp.ax_.set_title(f"{title} Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_overfitting_curve(X_train, y_train, X_test, y_test, max_depth_range=range(1, 21)):
    """Plots overfitting analysis by varying tree depth."""
    train_acc = []
    test_acc = []

    for depth in max_depth_range:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        train_acc.append(model.score(X_train, y_train))
        test_acc.append(model.score(X_test, y_test))

    plt.plot(max_depth_range, train_acc, label="Train Accuracy", marker='o')
    plt.plot(max_depth_range, test_acc, label="Test Accuracy", marker='o')
    plt.xlabel("Max Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Overfitting Analysis")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Feature Importance
def plot_feature_importance(model, feature_names):
    """Plots feature importance for a tree-based model."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances.sort_values().plot(kind='barh', figsize=(10, 6), color='mediumseagreen')
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# Cross-Validation
def cross_validate(model, X, y, cv=5):
    """Performs cross-validation and prints average score."""
    scores = cross_val_score(model, X, y, cv=cv)
    logger.info(f"Cross-validation scores: {scores}")
    logger.success(f"Average CV Accuracy: {scores.mean():.4f}")

# Main Execution
if __name__ == "__main__":
    df = load_data("HeartDiseaseDataset/heart.csv")
    explore_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # Train and Evaluate Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    evaluate_model(dt_model, X_test, y_test, title="Decision Tree")
    visualize_tree(dt_model, feature_names=X_train.columns, class_names=["No Disease", "Disease"])

    # Overfitting analysis
    plot_overfitting_curve(X_train, y_train, X_test, y_test)

    # Train and Evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, title="Random Forest")

    # Feature Importance
    plot_feature_importance(rf_model, X_train.columns)

    # Cross-validation
    cross_validate(rf_model, df.drop('target', axis=1), df['target'])
