# Titanic Dataset - Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
df = pd.read_csv("titanic_dataset/Titanic-Dataset.csv")


# Summary Statistics
print("Basic Info:")
print(df.info(), "\n")

print("Missing Values:")
print(df.isnull().sum(), "\n")

print("Summary Statistics (Numerical Columns):")
print(df.describe(), "\n")

print("Mean Values:")
print(df.mean(numeric_only=True), "\n")

print("Median Values:")
print(df.median(numeric_only=True), "\n")

print("Standard Deviation:")
print(df.std(numeric_only=True), "\n")

# Histograms
df.hist(bins=20, color='skyblue', edgecolor='black', figsize=(14, 10))
plt.suptitle('Histograms of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()

# Boxplots
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Boxplot of {col}')
    plt.show()

# Correlation Matrix & Pairplot
corr_matrix = df.corr(numeric_only=True)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']].dropna(), hue='Survived', palette='husl')
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()

# Identify Patterns, Trends, Anomalies
print("Survival Rate by Pclass:\n", df.groupby('Pclass')['Survived'].mean(), "\n")
print("Survival Rate by Sex:\n", df.groupby('Sex')['Survived'].mean(), "\n")
print("Average Fare by Pclass:\n", df.groupby('Pclass')['Fare'].mean(), "\n")

# 7. Inferences from Visuals
print("Inferences:")
print("* Females had a significantly higher survival rate than males.")
print("* Passengers in 1st class had better survival odds.")
print("* Fare is positively skewed with some high-value outliers.")
print("* Age distribution is fairly normal but has missing values.")
print("* Strong correlation between Fare and Pclass.")
print("* No strong multicollinearity found among numeric features.")
