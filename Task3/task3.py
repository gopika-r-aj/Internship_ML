# Linear Regression for House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("HousePriceDataset/Housing.csv")

# Map binary 'yes'/'no' columns to 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 
               'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Map furnishing status to ordinal integers
df['furnishingstatus'] = df['furnishingstatus'].map({
    'unfurnished': 0, 
    'semi-furnished': 1, 
    'furnished': 2
})

# Drop rows with any missing values
df.dropna(inplace=True)

# Feature-target split
X = df.drop('price', axis=1)
y = df['price']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display metrics
print("Evaluation Metrics:")
print(f"  MAE  : {mae:.2f}")
print(f"  MSE  : {mse:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")

# Coefficients
print("\nModel Coefficients:")
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print(coeff_df)

# Plot: Simple Linear Regression Line (e.g., 'area' vs 'price')
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['area'], y=df['price'], color='skyblue', label='Data Points')
simple_model = LinearRegression()
simple_model.fit(df[['area']], df['price'])
plt.plot(df['area'], simple_model.predict(df[['area']]), color='red', label='Regression Line')
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.tight_layout()
plt.show()


