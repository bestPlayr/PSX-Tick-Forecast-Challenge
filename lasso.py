import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('train.csv')

# Using Open_daily and LDCP directly, and other relevant columns
X = dataset.iloc[:, [1, 4, 5, 6, 7]].values  # Include Scrip, Volume, Date, Open_daily, LDCP
Y = dataset.iloc[:, 3].values  # Price

# DATA PREPROCESSING
X[:, 2] = pd.to_datetime(X[:, 2])

# Convert datetime to float (Unix timestamp)
X[:, 2] = [timestamp.timestamp() for timestamp in X[:, 2]]

# Converting categorical data to numerical data
scrip = ColumnTransformer([("Scrip", OneHotEncoder(), [0])], remainder='passthrough')
X = scrip.fit_transform(X)
X = X[:, 1:]  # Avoid dummy variable trap by removing the first dummy variable

# Convert the entire array to float
X = X.astype(float)

# Feature Scaling
sc_X = MinMaxScaler()
X_scaled = sc_X.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

# Initialize Lasso regressor
lasso_regressor = Lasso()

# Define parameter grid for Lasso
lasso_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'max_iter': [1000, 5000, 10000, 100000]
}

# Perform Grid Search with cross-validation for Lasso
lasso_grid_search = GridSearchCV(estimator=lasso_regressor, param_grid=lasso_param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train, y_train)

# Get best Lasso model
best_lasso_regressor = lasso_grid_search.best_estimator_

# Predict on test data
y_pred_lasso = best_lasso_regressor.predict(X_test)

# Evaluate Lasso model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape_score = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return mse, r2, smape_score

lasso_mse, lasso_r2, lasso_smape = evaluate_model(y_test, y_pred_lasso)

# Print evaluation results
print(f'Lasso Regression MSE: {lasso_mse}, R^2: {lasso_r2}, SMAPE: {lasso_smape}')

# Plot true vs predicted prices
plt.figure(figsize=(10, 5))
plt.plot(y_test[1000:1200], color='blue', label='Actual Prices')
plt.plot(y_pred_lasso[1000:1200], color='red', label='Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Samples')
plt.ylabel('Price')
plt.legend()
plt.show()

# Load test dataset
test_dataset = pd.read_csv('test.csv')

# Using Open_daily and LDCP directly
X2_test = test_dataset.iloc[:, [1, 3, 4, 5, 6]].values  # Adjust column indices accordingly

# DATA PREPROCESSING
X2_test[:, 2] = pd.to_datetime(X2_test[:, 2])

# Convert datetime to float (Unix timestamp)
X2_test[:, 2] = [timestamp.timestamp() for timestamp in X2_test[:, 2]]

# Converting categorical data to numerical data
X2_test = scrip.transform(X2_test)
X2_test = X2_test[:, 1:]  # Avoid dummy variable trap by removing the first dummy variable

# Convert the entire array to float
X2_test = X2_test.astype(float)

# Feature scaling
X2_test = sc_X.transform(X2_test)

# Predicting prices for the test data using the best Lasso model
y_pred_test = best_lasso_regressor.predict(X2_test)

y_pred_test = y_pred_test.flatten()

# Printing predicted prices for the test data along with IDs
ids = test_dataset.iloc[:, 0].values  # Assuming the first column contains IDs

# Create a DataFrame with IDs and predicted prices
results_df = pd.DataFrame({'ID': ids, 'Price': y_pred_test})

# Save the DataFrame to a CSV file
results_df.to_csv('lasso_2.csv', index=False)
print("Predicted prices saved")
