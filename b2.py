import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Fetch the Boston Housing dataset from OpenML
boston_data = fetch_openml(name='boston', version=1)
X = boston_data.data
y = boston_data.target
feature_names = boston_data.feature_names

# Convert the dataset into a pandas DataFrame for easier manipulation
df = pd.DataFrame(X, columns=feature_names)
df['Price'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for Ridge Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models: Linear Regression and Ridge Regression
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)  # Experiment with alpha parameter for Ridge

# Train both models
linear_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)

# Predict house prices on the test set using both models
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluate the models using Mean Squared Error (MSE) and R² Score
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

r2_linear = r2_score(y_test, y_pred_linear)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Print evaluation results
print("Linear Regression Model:")
print(f"Mean Squared Error (MSE): {mse_linear}")
print(f"R² Score: {r2_linear}\n")

print("Ridge Regression Model:")
print(f"Mean Squared Error (MSE): {mse_ridge}")
print(f"R² Score: {r2_ridge}\n")

# Plotting: Actual vs Predicted Prices for both models

plt.figure(figsize=(14, 6))

# Linear Regression: Actual vs Predicted Prices
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, color='blue', alpha=0.6, label="Linear Regression")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
plt.title('Linear Regression: Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()

# Ridge Regression: Actual vs Predicted Prices
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.6, label="Ridge Regression")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
plt.title('Ridge Regression: Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

# Feature Importance Analysis (for Ridge Regression)
coefficients = ridge_model.coef_
feature_importance = pd.DataFrame(coefficients, index=feature_names, columns=["Coefficient"])
print("Feature Importance (Ridge Regression):")
print(feature_importance)
