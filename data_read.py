import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from helper_functions import *

columns = ["TimeStamp", "MoveType", "AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]
datacolumns = ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]

def plot_windows():
    """
    Plots sliding windows of a specified feature from preprocessed data.
    """
    # Clean the data and normalize
    X, y = preprocess_data(df, datacolumns)
    
    #  1            2           3           4           5        6        7         8         9
    # ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]
    n = 8
    graph = X[:, n]
    
    # Find the peaks in the data to calulate the window size
    # Interval sets the minimum distance between each peak
    interval = 100
    windows, minima, local_minima = local_Minima(X, interval)
    
    # Set the values for the window size and step size
    window_size = windows
    step_size = windows
    print("The window size used: " + str(window_size))

    # Split the data into sliding windows
    windows_X, windows_y = sliding_window(X, y, window_size, step_size)
    
    # create a new figure
    fig, axs = plt.subplots(2, layout='constrained')
    
    # Plotting graph data
    axs[0].plot(graph, c='g')
    axs[0].set_title(datacolumns[n - 1])
    
    m = window_size
    x = np.arange(m).reshape(-1, 1)
    k = 0
    # loop over the sliding windows
    for i, window in enumerate(windows_X):
        # extract the feature data of interest from the window
        feature_data = window[:, 8]
        
        # plot the feature data
        axs[1].plot(x, feature_data, label=f'Sliding window {i}')

        m += window_size
        x = (np.arange(x[-1] + 1, m).reshape(-1, 1)) 
    
    # Plot the data split into windows
    axs[1].scatter(minima, local_minima, c='b')
        
    # add labels and title
    axs[1].set_title('Feature data for sliding windows')

    # display the plot
    plt.show()

def plot_data(data_required):
    """
    Plots three graphs of the specified data required from preprocessed data.

    Parameters:
        data_required: The index of the first feature of interest in the preprocessed data,
        with valid values ranging from 0 to 2 inclusive.
    """
    # Clean the data and normalize
    X, y = preprocess_data(df, datacolumns)
    
    i = 1 + (3 * data_required)
    
    # Extract the data required for each graph
    graph1 = X[:, i]
    graph2 = X[:, i + 1]
    graph3 = X[:, i + 2]
    
    # Plot each graph with the axes labelled
    fig, axs = plt.subplots(3, layout='constrained')
    axs[0].plot(graph1)
    axs[0].set_title(datacolumns[i  -1])
    axs[1].plot(graph2, c='g')
    axs[1].set_title(datacolumns[i])
    axs[2].plot(graph3, c='r')
    axs[2].set_title(datacolumns[i + 1])
    plt.show()

# Data to be displayed on the graph
df = pd.read_csv('UnseenData/test2.csv')

data = df
df = df[columns]

# Plot the data split into windows
plot_windows()

# Plot the data of the specified sensor
# data_required = 0 - acelerometer data, = 1 - gyroscope data, = 2 - gravity data
data_required = 2
plot_data(data_required)
