# task1.py
 
# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('titanic_dataset/Titanic-Dataset.csv')
print("Initial data:")
print(df.head())

# Show basic info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  

# Encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Detect outliers using boxplot
plt.figure(figsize=(10,5))
sns.boxplot(x=df['Fare'])
plt.title('Fare Outliers')
plt.savefig("fare_boxplot.png")  
plt.close()

# Remove outliers (based on IQR for Fare)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]

# Save cleaned data
df.to_csv("cleaned_titanic.csv", index=False)
print("\n Data preprocessing completed. Cleaned data saved as 'cleaned_titanic.csv'.")
