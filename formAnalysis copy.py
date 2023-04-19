import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  

from sklearn.ensemble import RandomForestClassifier

training_file = "squatData"
testing_file = "10repBadform"

df = pd.read_csv(training_file +'.csv')

columns = ["TimeStamp", "MoveType", "AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]
datacolumns = ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]

df = df[columns]

testdf = pd.read_csv('GoodData/' + testing_file +'.csv')

testdf = testdf[columns]

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

def local_minima(X, interval):
    # Attempt to make the windows more even
    DMG_feature = np.transpose(X)
    feature8 = DMG_feature[8]
    
    # Find local minima
    minima = argrelextrema(-feature8, np.less_equal, order=interval)[0]

    print(len(minima))

    local_minima = []
    # print the local minima
    for min in minima:
        local_minima.append(feature8[min])
    
    windows = int( len(feature8) / len(minima) )
    return windows

def train_model(X, y, window_size, step_size):
    # Split the data into sliding windows
    windows_X, windows_y = sliding_window(X, y, window_size, step_size)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(windows_X, windows_y, test_size=0.2, random_state=42)
    
    # Create and fit the random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    
    # Make predictions on the testing data
    y_pred = clf.predict(X_test.reshape(X_test.shape[0], -1))
    
    # Calculate the accuracy of the model
    acc = accuracy_score(y_test, y_pred)
    return acc, clf

def test_model(clf, X_new, y_new, window_size, step_size):
    # Split the data into sliding windows
    windows_X_new, windows_y_new = sliding_window(X_new, y_new, window_size, step_size)

    # Make predictions on the testing data
    y_pred_new = clf.predict(windows_X_new.reshape(windows_X_new.shape[0], -1))
    
    print(y_pred_new)

    # Calculate the accuracy of the model
    acc_new = accuracy_score(windows_y_new, y_pred_new)
    
    return acc_new

# Clean the data and normalize
X, y = clean_data(df, datacolumns)

# Set the interval for finding local minima
interval = 140
windows = local_minima(X, interval)

window_size = windows
step_size = windows

acc, clf = train_model(X, y, window_size, step_size)
print("Accuracy against trained data: {:.2f}%".format(acc * 100))

# Clean the data and normalize
X_new, y_new = clean_data(testdf, datacolumns)
acc_new = test_model(clf, X_new, y_new, window_size, step_size)


print("Accuracy on new data: {:.2f}%".format(acc_new * 100))