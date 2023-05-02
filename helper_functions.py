import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema

def preprocess_data(df, columns, threshold=3):
    """
    Preprocesses the data by removing missing values and outliers, normalizing the features using 
    a StandardScaler, and separating the target column from the features.

    Parameters:
        df: pandas DataFrame containing the data to be preprocessed
        columns: list of column names to be preprocessed
        threshold: number of standard deviations from the mean beyond which a row is considered an outlier 
                   (default=3)
    Return:
        X: numpy array of preprocessed features
        y: pandas Series of the target column
    """
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

def sliding_window(X, y, window_size, step_size):
    """
    Split the input data into sliding windows of size 'window_size' and step size 'step_size'.

    Parameters:
        X: An array (n_samples, n_features) containing the input data features.
        y: An array (n_samples) containing the input data labels.
        window_size: The size of the sliding window.
        step_size: The step size for the sliding window.

    Returns:
        windows_X: A 3D array (n_windows, window_size, n_features) containing the input data
                   split into sliding windows.
        windows_y: An array (n_windows) containing the corresponding labels for each sliding window.
    """ 
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
    """
    Function to find peaks in the data

    Parameters:
        X: An array of features
        interval: The minimum distance between peaks
    Returns:
        windows: The window size for the data to be split evenly
        minima: The array of peaks detected 
        local_minima: The array of values of the peaks detected
    """ 
    
    DMG_feature = np.transpose(X)
    feature8 = DMG_feature[8]
    
    # Find local minima
    minima = argrelextrema(-feature8, np.less_equal, order=interval)[0]

    local_minima = []
    for min in minima:
        local_minima.append(feature8[min])
    
    # Calculate the window size
    windows = int( len(feature8) / len(minima) )
    return windows, minima, local_minima
