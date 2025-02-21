#©2025[faika].licensed underCC-BY-NC-ND 4.0.
# Unauthorized commercial use or modification is prohibited
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Generate synthetic biomarker and lifestyle data for 90 days
def generate_synthetic_data(days=90):
    # Define date range
    start_date = datetime.today() - timedelta(days=days)
    date_range = pd.date_range(start=start_date, periods=days, freq='D')

    # Define realistic ranges for biomarkers and lifestyle factors
    glucose_range = (70, 140)  # mg/dL
    cortisol_range = (5, 25)  # mcg/dL
    estrogen_range = (15, 350)  # pg/mL
    progesterone_range = (0.1, 20)  # ng/mL
    sleep_hours_range = (4, 9)  # hours
    steps_range = (1000, 10000)  # steps per day
    stress_level_range = (1, 10)  # scale of 1-10

    # Generate random data within the defined ranges
    data = {
        "Date": date_range,
        "Glucose (mg/dL)": np.random.uniform(glucose_range[0], glucose_range[1], days),
        "Cortisol (mcg/dL)": np.random.uniform(cortisol_range[0], cortisol_range[1], days),
        "Estrogen (pg/mL)": np.random.uniform(estrogen_range[0], estrogen_range[1], days),
        "Progesterone (ng/mL)": np.random.uniform(progesterone_range[0], progesterone_range[1], days),
        "Sleep (hours)": np.random.uniform(sleep_hours_range[0], sleep_hours_range[1], days),
        "Steps": np.random.randint(steps_range[0], steps_range[1], days),
        "Stress Level": np.random.randint(stress_level_range[0], stress_level_range[1], days),
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

# Generate the data
synthetic_data = generate_synthetic_data()

# Save the data to a CSV file
synthetic_data.to_csv("synthetic_biomarker_lifestyle_data.csv", index=False)

# Display the first few rows of the data
print(synthetic_data.head())

# Step 2: Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(synthetic_data.iloc[:, 1:])  # Exclude the 'Date' column

# Step 3: Prepare the data for LSTM
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(scaled_data, seq_length)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 4: Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, X_train.shape[2])),
    Dense(X_train.shape[2])
])
model.compile(optimizer='adam', loss='mse')

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)

# Step 6: Predict the next 10 days
def predict_future(model, data, steps=10):
    predictions = []
    last_sequence = data[-1]
    for _ in range(steps):
        next_pred = model.predict(last_sequence[np.newaxis, :, :])
        predictions.append(next_pred[0])
        last_sequence = np.vstack([last_sequence[1:], next_pred])
    return np.array(predictions)

future_predictions = predict_future(model, X_test)

# Denormalize the predictions
future_predictions_denorm = scaler.inverse_transform(future_predictions)

# Step 7: Visualize the results
# Plot actual biomarker and lifestyle factors over time
plt.figure(figsize=(15, 10))
for i, col in enumerate(synthetic_data.columns[1:]):
    plt.subplot(3, 3, i+1)
    plt.plot(synthetic_data['Date'], synthetic_data[col], label=col)
    plt.title(col)
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.legend()
plt.tight_layout()
plt.show()

# Plot actual vs predicted glucose levels
plt.figure(figsize=(10, 6))
plt.plot(synthetic_data['Date'][-len(y_test):], scaler.inverse_transform(y_test)[:, 0], label='Actual Glucose')
plt.plot(pd.date_range(start=synthetic_data['Date'].iloc[-1] + timedelta(days=1), periods=10, freq='D'),
         future_predictions_denorm[:, 0], label='Predicted Glucose', linestyle='--')
plt.title('Actual vs Predicted Glucose Levels')
plt.xlabel('Date')
plt.ylabel('Glucose (mg/dL)')
plt.legend()
plt.show()
