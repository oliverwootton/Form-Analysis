import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema

# Preprocess that data
def clean_data(df, columns, threshold=3):
    # Remove rows that contain missing values
    df = df.dropna()
    
    # Iterate through each column in 'columns'
    for column in columns:
        # Calculate the mean and standard deviation of the column's values
        mean = df[column].mean()
        std = df[column].std()
        
        # Remove any rows whose values in this column are more than 'threshold' standard deviations 
        # away from the mean, effectively removing outliers
        df = df[np.abs(df[column] - mean) <= (threshold * std)]

    # Separate the target column (MoveType) from the features
    X = df.drop("MoveType", axis=1)
    y = df["MoveType"]

    # Normalize the data using a StandardScaler object from scikit-learn
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Return the processed data as a tuple of X and y
    return X, y

# Split the data into sliding windows of size window_size and step size step_size
def sliding_window(X, y, window_size, step_size):
    windows_X = []
    windows_y = []
    n_samples = X.shape[0]
    for i in range(0, n_samples - window_size + 1, step_size):
        window_X = X[i:i+window_size, :]
        window_y = y[i:i+window_size]
        windows_X.append(window_X)
        windows_y.append(window_y.iloc[0])
    windows_X = np.array(windows_X)
    windows_y = np.array(windows_y)
    return windows_X, windows_y

def local_Minima(X, interval):
    # Attempt to make the windows more even
    DMG_feature = np.transpose(X)
    feature8 = DMG_feature[8]
    
    # Find local minima
    minima = argrelextrema(-feature8, np.less_equal, order=interval)[0]

    local_minima = []
    # print the local minima
    for min in minima:
        local_minima.append(feature8[min])
    
    # Calculate the window size
    windows = int( len(feature8) / len(minima) )
    return windows, minima, local_minima

