# Task 3 – House Price Prediction using Linear Regression

  The goal is to implement both **simple and multiple linear regression** to predict house prices based on various features in a real-world housing dataset.

## Objective

- Understand and implement **Linear Regression** (both simple and multiple).
- Preprocess real-world data (handle categorical variables and missing values).
- Train and evaluate a regression model using proper metrics.
- Visualize the regression line for interpretation.
- Interpret model coefficients to understand feature impacts.


## Dataset Used

- Source: [House Price Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- Features:
  - `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
  - Binary fields: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
  - Categorical field: `furnishingstatus`
- Target:
  - `price` – House price (numeric)


## Technologies Used

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## Steps Performed

1. **Loaded Dataset** from CSV
2. **Preprocessed Data**:
   - Encoded categorical columns (`yes/no` → `1/0`, furnishing status → ordinal)
   - Handled missing values using `dropna()`
3. **Split Data** into training and testing sets (80/20)
4. **Trained Linear Regression Model** using `sklearn.linear_model`
5. **Evaluated Model** with:
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - R² Score (Coefficient of Determination)
6. **Visualized** the regression line for `area` vs `price`
7. **Displayed Model Coefficients** for interpretation


## Outputs

 The screenshots of the outputs are added to "Output_screenshots"


