import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

class Data:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
  
    def xData(self):
        return [row[1]["AccelroX"] for row in (self.df).iterrows()]
                
    def yData(self):
        return [row[1]["AceelroY"] for row in (self.df).iterrows()]

    def zData(self):
        return [row[1]["AceelroZ"] for row in (self.df).iterrows()]

    def recNo(self):
        return [row[1]["RecNo"] for row in (self.df).iterrows()]
    
    
    
def yData(df):
        return [row[1]["AceelroY"] for row in (df).iterrows()]

df = pd.read_csv('Data/u01_movements_wr-su_d_rt_100_20230118-1502.csv')

# Remove any missing or outlier data
df = df.dropna()
df = df[np.abs(df["AccelroX"] - df["AccelroX"].mean())<=(3*df["AccelroX"].std())]
df = df[np.abs(df["AceelroY"] - df["AceelroY"].mean())<=(3*df["AceelroY"].std())]
df = df[np.abs(df["AceelroZ"] - df["AceelroZ"].mean())<=(3*df["AceelroZ"].std())]

# Normalise the data
scaler = preprocessing.StandardScaler()
df[["AccelroX", "AceelroY", "AceelroZ"]] = scaler.fit_transform(df[["AccelroX", "AceelroY", "AceelroZ"]])

# # Split the data into windows
# window_size = 505
# stride = 450
# num_windows = int((len(df) - window_size) / stride) + 1
# windows = [df[i:i + window_size] for i in range(0, num_windows * stride, stride)]

# # Label the windows
# labels = [df["MoveType"][i + window_size // 2] for i in range(0, num_windows * stride, stride)]

# # Split the windows into training and testing sets
# train_windows = windows[:int(0.8 * num_windows)]
# train_labels = labels[:int(0.8 * num_windows)]
# test_windows = windows[int(0.8 * num_windows):]
# test_labels = labels[int(0.8 * num_windows):]

# # Create the plot
# for window, label in zip(train_windows, train_labels):
#     plt.plot(window["AccelroX"], label = label)
# plt.xlabel("Time")
# plt.ylabel("Accelerometer X")
# plt.legend()
# plt.show()

plt.plot(yData(df), 'm', linestyle = 'dotted')
        
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.show()