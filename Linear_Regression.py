import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [1, 4, 5, 6, 7]].values
Y = dataset.iloc[:, 3].values

X[:,2] = pd.to_datetime(X[:,2])

X[:, 2] = [timestamp.timestamp() for timestamp in X[:, 2]]

scrip = ColumnTransformer([("Scrip", OneHotEncoder(), [0])], remainder='passthrough')
X = scrip.fit_transform(X)
X = X[: , 0:]

X = X.astype(float)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.153, random_state = 0  )

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


R_square = regressor.score(X_test , y_test)
print(R_square)

test_dataset = pd.read_csv('test.csv')
X2_test = test_dataset.iloc[:, [1, 3, 4, 5, 6]].values

X2_test[:, 2] = pd.to_datetime(X2_test[:, 2])

X2_test[:, 2] = [timestamp.timestamp() for timestamp in X2_test[:, 2]]

X2_test = scrip.fit_transform(X2_test)

X2_test = X2_test.astype(float)

y_pred_test = regressor.predict(X2_test)

ids = test_dataset.iloc[:, 0].values

results_df = pd.DataFrame({'ID': ids, 'Price': y_pred_test})

results_df.to_csv('predicted_test_prices.csv', index=False)

print("Predicted prices saved to predicted_prices.csv")