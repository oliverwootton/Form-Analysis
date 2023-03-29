import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier    

training_file = "MissMatchedWeight2"

df = pd.read_csv("NewData/" + training_file +'.csv')

columns = ["TimeStamp", "MoveType", "AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]
datacolumns = ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]

data = df

# Find maximum and min Peaks for rep counting
n = 150
# data['min'] = data.iloc[argrelextrema(data.DMGrvY.values, np.less_equal, order=n)[0]]['DMGrvY']
# data['max'] = data.iloc[argrelextrema(data.DMGrvY.values, np.greater_equal, order=n)[0]]['DMGrvY']

fig, axs = plt.subplots(3, layout='constrained')
fig.suptitle("DMGrv Data")
# axs[0].scatter(data.index, data['max'], c='g')
# axs[0].scatter(data.index, data['min'], c='r')
axs[0].plot(data.index, data['DMGrvX'])
axs[0].set_title("DMGrvX")

axs[1].plot(data.index, data['DMGrvY'], c='g')
axs[1].set_title("DMGrvY")

axs[2].plot(data.index, data['DMGrvZ'], c='r')
axs[2].set_title("DMGrvZ")
# fig.xlabel("Time (s)")
# fig.ylabel("Accelerometer Y (m/s^2)")
plt.show()