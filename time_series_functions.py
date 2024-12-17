# Main functions to help improve readability of Time Series Notebook

# Import Libraries
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

# Create LSTM dataset
def create_lstm_dataset(data, lookback):
    # scale data to between 0 and 1
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])

    return np.array(X), np.array(y)


# Split Time Series Data
def split_time_series(X, y, train_frac=0.7, val_frac=0.15):
    """
    Splits time-series data into train, validation, and test sets.

    Parameters:
    X (pd.DataFrame or np.array): Features (input data).
    y (pd.Series or np.array): Target (output data).
    train_frac (float): Fraction of data to use for training.
    val_frac (float): Fraction of data to use for validation.

    Returns:
    tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert len(X) == len(y), "X and y must have the same length"

    train_size = int(len(X) * train_frac)
    val_size = int(len(X) * val_frac)

    # Split features (X)
    X_train = X[:train_size]
    X_val = X[train_size : train_size + val_size]
    X_test = X[train_size + val_size :]

    # Split target (y)
    y_train = y[:train_size]
    y_val = y[train_size : train_size + val_size]
    y_test = y[train_size + val_size :]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Plot Predictions
def plot_predictions(model, X, y, start=0, end=100):
    predictions = model.predict(X).flatten()
    df = pd.DataFrame(data={"Predictions": predictions, "Actual": y})
    plt.plot(df["Predictions"][start:end])
    plt.plot(df["Actual"][start:end])

    return df, mse(y, predictions)
