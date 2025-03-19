# Boston Housing Price Prediction

This Python application predicts housing prices using Linear Regression and Ridge Regression. The model is trained on the Boston Housing dataset and evaluates performance using various metrics. It also includes data visualization for better insights.

## Features

- **Predict House Prices**: Uses Linear and Ridge Regression models to estimate housing prices.
- **Data Preprocessing**: Standardizes features for improved model performance.
- **Model Evaluation**: Provides Mean Squared Error (MSE) and R² score for performance assessment.
- **Feature Importance Analysis**: Identifies the most influential features in house price prediction.
- **Data Visualization**: Includes scatter plots comparing actual vs predicted prices.

## Requirements

### Prerequisites

- **Python 3.x**
- Install required dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/boston-housing-price-prediction.git
cd boston-housing-price-prediction
```

2. Run the application:

```bash
python housing_price_prediction.py
```

## Code Structure

### **Dataset Loading**
- Fetches the Boston Housing dataset from OpenML.
- Converts it into a pandas DataFrame for easier manipulation.

### **Preprocessing**
- Splits the dataset into training and testing sets.
- Standardizes features using `StandardScaler` for better Ridge Regression performance.

### **Model Training**
- Trains both Linear Regression and Ridge Regression models.
- Ridge Regression includes an alpha parameter for regularization.

### **Model Evaluation**
- Computes Mean Squared Error (MSE) and R² score for both models.
- Visualizes actual vs predicted prices using scatter plots.

### **Feature Importance**
- Displays Ridge Regression coefficients to analyze feature impact on house prices.

## Example Workflow

1. **Dataset Loading**: Fetches and prepares the Boston Housing dataset.
2. **Preprocessing**: Splits data and standardizes features.
3. **Model Training**: Trains Linear and Ridge Regression models.
4. **Prediction & Evaluation**: Evaluates models using MSE and R² score.
5. **Visualization**: Plots actual vs predicted prices and feature importance.

## Example Output

- **Mean Squared Error (MSE)**: Displays error between predicted and actual prices.
- **R² Score**: Measures model performance.
- **Scatter Plots**: Visual representation of predictions.
- **Feature Importance**: Shows the impact of different housing features on price.

## Screenshots

(Add plots comparing actual vs predicted prices here)

## Troubleshooting

- **High Error**: Ensure proper data preprocessing and feature scaling.
- **Poor Predictions**: Experiment with Ridge Regression's alpha parameter.
- **Dataset Issues**: Verify that the dataset is correctly loaded from OpenML.

## License

This project is licensed under the MIT License.

---

Developed by [Your Name](https://github.com/yourusername)

