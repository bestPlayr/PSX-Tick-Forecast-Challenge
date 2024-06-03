import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib as plt
from tensorflow.keras.optimizers import Adam

# Importing the dataset
dataset = pd.read_csv('train.csv')

dataset['AVG'] = (dataset['Open_daily'] + dataset['LDCP']) / 2
dataset.drop(['Open_daily', 'LDCP'], axis=1, inplace=True)

X = dataset.iloc[:, [1, 4, 5, 6]].values
Y = dataset.iloc[:, 3].values

# DATA PREPROCESSING
X[:, 2] = pd.to_datetime(X[:, 2])

# Convert datetime to float (Unix timestamp)
X[:, 2] = [timestamp.timestamp() for timestamp in X[:, 2]]

# Converting categorical data to numerical data
scrip = ColumnTransformer([("Scrip", OneHotEncoder(), [0])], remainder='passthrough')
X = scrip.fit_transform(X)
X = X[:, 0:]

# Convert the entire array to float
X = X.astype(float)

# Feature Scaling
sc_X = MinMaxScaler()
X_scaled = sc_X.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.0001, random_state=0)

# Reshape for GRU
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# IMPORTING MODELS AND LAYERS
# Making the GRU
regressor = Sequential()

# Adding input layer
regressor.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding first layer
regressor.add(GRU(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding second layer
regressor.add(GRU(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(GRU(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=50, activation='relu'))

# Output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Fitting the regressor to the data
history = regressor.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stopping, model_checkpoint])

# Evaluating the model
mse = regressor.evaluate(X_test, y_test)
print('Mean Squared Error:', mse)

# Predicting
predicted_prices = regressor.predict(X_test)

# Inverse transform the predicted prices
# predicted_prices = sc_y.inverse_transform(predicted_prices)
# y_test_inverse = sc_y.inverse_transform(y_test)

# Load test dataset
test_dataset = pd.read_csv('test.csv')
test_dataset['AVG'] = (test_dataset['Open_daily'] + test_dataset['LDCP']) / 2
test_dataset.drop(['Open_daily', 'LDCP'], axis=1, inplace=True)

X2_test = test_dataset.iloc[:, [1, 3, 4, 5]].values  # Adjust column indices since there's no Price column

# DATA PREPROCESSING
X2_test[:, 2] = pd.to_datetime(X2_test[:, 2])

# Convert datetime to float (Unix timestamp)
X2_test[:, 2] = [timestamp.timestamp() for timestamp in X2_test[:, 2]]

# Converting alphabetical data to numerical data
X2_test = scrip.fit_transform(X2_test)
# X2_test = X2_test[:, 1:]  # Remove the first dummy variable to avoid dummy variable trap

# Convert the entire array to float
X2_test = X2_test.astype(float)

X2_test = sc_X.transform(X2_test)
X2_test = np.reshape(X2_test, (X2_test.shape[0], X2_test.shape[1], 1))

# Predicting prices for the test data
y_pred_test = regressor.predict(X2_test)

y_pred_test = y_pred_test.flatten()

# Printing predicted prices for the test data along with IDs
ids = test_dataset.iloc[:, 0].values  # Assuming the first column contains IDs

# Create a DataFrame with IDs and predicted prices
results_df = pd.DataFrame({'ID': ids, 'Price': y_pred_test})

# Save the DataFrame to a CSV file
results_df.to_csv('gru_avg.csv', index=False)