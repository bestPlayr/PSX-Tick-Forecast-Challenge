import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load and preprocess the training data
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [1, 4, 5, 6, 7]].values
y = dataset.iloc[:, 3].values

# Convert timestamp to numeric value
X[:, 2] = pd.to_datetime(X[:, 2])
X[:, 2] = [timestamp.timestamp() for timestamp in X[:, 2]]

# One-hot encode the first column
scrip = ColumnTransformer([("Scrip", OneHotEncoder(), [0])], remainder='passthrough')
X = scrip.fit_transform(X)
X = X[:, 1:]  # Drop the first column to avoid dummy variable trap

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.001, random_state=0)

# Define the LSTM model
def create_lstm_model(input_shape, units_1, units_2):
    model = Sequential()
    model.add(LSTM(units=units_1, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=units_2, activation='relu'))
    model.add(Dense(units=1))
    return model

# Reshape the input for LSTM model
input_shape = (X_train.shape[1], 1)  # Shape of input for LSTM
model = create_lstm_model(input_shape, 16, 32)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=3500, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Plot training and validation loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plotting for testing data
def plot_predicted_vs_actual(y_actual, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, color='red', label='Actual')
    plt.plot(y_pred, color='blue', label='Predicted')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

plot_predicted_vs_actual(y_test, y_pred)

# Load the test data
test_dataset = pd.read_csv('test.csv')

# Extract features from test data
X_test_data = test_dataset.iloc[:, [1, 3, 4, 5, 6]].values

# Convert timestamp to numeric value
X_test_data[:, 2] = pd.to_datetime(X_test_data[:, 2])
X_test_data[:, 2] = [timestamp.timestamp() for timestamp in X_test_data[:, 2]]

# One-hot encode the first column
X_test_data = scrip.transform(X_test_data)
X_test_data = X_test_data[:, 1:]  # Drop the first column to avoid dummy variable trap

# Scale the features using the same scaler used for training data
X_test_scaled = scaler.transform(X_test_data)

# Reshape the input for LSTM model
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Make predictions using the trained LSTM model
y_pred_test = model.predict(X_test_scaled)

# Save the predictions to a CSV file
ids = test_dataset.iloc[:, 0].values
results_df = pd.DataFrame({'ID': ids, 'Price': y_pred_test.flatten()})
results_df.to_csv('LSTM_3500.csv', index=False)

print("Predicted prices saved to LSTM_100.csv")

model.save('lstm_3500_epoch.h5')
model.save('lstm_3500_epoch.keras')
