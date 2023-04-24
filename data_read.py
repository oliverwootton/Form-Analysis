import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  

training_file = "AllData"

# df = pd.read_csv('GoodData/3.csv')
# df = pd.read_csv(training_file +'.csv')
df = pd.read_csv('test2.csv')

columns = ["TimeStamp", "MoveType", "AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]
datacolumns = ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]

data = df

df = df[columns]

def clean_data(df, columns, threshold=3):
    df = df.dropna()
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        df = df[np.abs(df[column] - mean) <= (threshold * std)]

    # separate the target column (MoveType) from the features
    X = df.drop("MoveType", axis=1)
    y = df["MoveType"]

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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
    
    windows = int( len(feature8) / len(minima) )
    return windows, minima, local_minima

def plot_windows():
    # Clean the data and normalize
    X, y = clean_data(df, datacolumns)
    
    DMGrvY = X[:, 8]
    
    interval = 140
    windows, minima, local_minima = local_Minima(X, interval)
    
    print(windows)
    
    window_size = windows
    step_size = windows

    # Split the data into sliding windows
    windows_X, windows_y = sliding_window(X, y, window_size, step_size)
    
    # create a new figure
    fig, axs = plt.subplots(2, layout='constrained')
    
    axs[0].plot(DMGrvY, c='g')
    axs[0].set_title("DMGrvY")
    
    m = window_size
    x = np.arange(m).reshape(-1, 1)
    k = 0
    # loop over the sliding windows
    for i, window in enumerate(windows_X):
        # extract the feature data of interest from the window
        feature_data = window[:, 8]
        
        # plot the feature data
        axs[1].plot(x, feature_data, label=f'Sliding window {i}')
        
        # ax.scatter(x[-1], connected_data(local_minima[k]))
        
        m += window_size
        x = (np.arange(x[-1] + 1, m).reshape(-1, 1)) 
    
    # ax.scatter(0, connected_data[0], c='b')
    axs[1].scatter(minima, local_minima, c='b')
    # ax.plot(connected_data)
        
    # add labels and title
    axs[1].set_title('Feature data for all sliding windows')

    # display the plot
    plt.show()

# ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]

def plot_data():
    # Clean the data and normalize
    X, y = clean_data(df, datacolumns)
    
    DMGrvX = X[:, 7]
    DMGrvY = X[:, 8]
    DMGrvZ = X[:, 9]
    
    fig, axs = plt.subplots(3, layout='constrained')
    fig.suptitle("DMGrvX")
    axs[0].plot(DMGrvX)
    axs[0].set_title("DMGrvX")

    axs[1].plot(DMGrvY, c='g')
    axs[1].set_title("DMGrvY")

    axs[2].plot(DMGrvZ, c='r')
    axs[2].set_title("DMGrvZ")
    plt.show()
      

plot_windows()
# plot_data()
