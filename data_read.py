import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

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
    
def plot(windows, column):
    for window in windows:
        plt.plot(window[column])
    plt.xlabel("Time (s)")
    plt.ylabel("Accelerometer Y (m/s^2)")
    plt.show()

def plot2(windows, labels, column):
    for window, label in zip(windows, labels):
        plt.plot(window[column], label = label)
    plt.xlabel("Time (s)")
    plt.ylabel("Accelerometer Y (m/s^2)")
    plt.legend()
    plt.show()
        

df = pd.read_csv('Data/u02_movements_el-ex_nd_lt_100_20230215-1533.csv')

"""

u01_movements_el-ex_nd_lt_100_20230215-1527
u01_movements_el-ex_nd_lt_100_20230215-1528

u02_movements_el-ex_nd_lt_100_20230215-1532
u02_movements_el-ex_nd_lt_100_20230215-1533
u02_movements_el-ex_nd_lt_100_20230215-1535
u02_movements_el-ex_nd_lt_100_20230215-1536
u02_movements_el-ex_nd_lt_100_20230215-1541
u02_movements_el-ex_nd_lt_100_20230215-1542
u02_movements_el-ex_nd_lt_100_20230215-1545
u02_movements_el-ex_nd_lt_100_20230215-1549
u02_movements_el-ex_nd_lt_100_20230215-1554
u02_movements_el-ex_nd_lt_100_20230215-1556

"""

# # Remove any missing or outlier data
# df = df.dropna()
# df = df[np.abs(df["AccelroX"] - df["AccelroX"].mean())<=(3*df["AccelroX"].std())]
# df = df[np.abs(df["AceelroY"] - df["AceelroY"].mean())<=(3*df["AceelroY"].std())]
# df = df[np.abs(df["AceelroZ"] - df["AceelroZ"].mean())<=(3*df["AceelroZ"].std())]
# df = df[np.abs(df["DMGrvY"] - df["DMGrvY"].mean())<=(3*df["DMGrvY"].std())]

# # Normalise the data
# scaler = preprocessing.StandardScaler()
# df[["AccelroX", "AceelroY", "AceelroZ", "DMGrvY"]] = scaler.fit_transform(df[["AccelroX", "AceelroY", "AceelroZ", "DMGrvY"]])


# data = df.loc[int(0.25 * len(df)):int(0.75 * len(df))]
data = df


n = 150
data['min'] = data.iloc[argrelextrema(data.DMGrvY.values, np.less_equal, order=n)[0]]['DMGrvY']
data['max'] = data.iloc[argrelextrema(data.DMGrvY.values, np.greater_equal, order=n)[0]]['DMGrvY']


plt.scatter(data.index, data['max'], c='g')
plt.scatter(data.index, data['min'], c='r')
plt.plot(data.index, data['DMGrvY'])
plt.show()


# # Create the plot with labeled windows
# plot2(train_windows, train_labels, "DMGrvY")

# Create the plot of just AceelroY data
# plt.plot(df[["AccelroY"]], 'm', linestyle = 'dotted')
        
# plt.xlabel("Time (s)")
# plt.ylabel("Acceleration (m/s^2)")
# plt.show()

# print(windows[5])
# print(df.loc[1022])