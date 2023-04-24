import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.model_selection import train_test_split

# from keras.models import Sequential
# from keras.layers import Dense, LSTM

training_file = "AllData"


df = pd.read_csv(training_file +'.csv')

columns = ["TimeStamp", "MoveType", "AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]
datacolumns = ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]

df = df[columns]

testdf = pd.read_csv('test2.csv')

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

    local_minima = []
    # print the local minima
    for min in minima:
        local_minima.append(feature8[min])
    
    print("Number of reps: " + str(len(minima)))
    
    windows = int( len(feature8) / len(minima) )
    return windows

def rF(X_train, y_train):
    # Create and fit the random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def sVC(X_train, y_train):
    # Create and fit the SVM classifier
    model = SVC(kernel='linear', C=1, gamma='auto', random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def kN(X_train, y_train):
    # Create and fit the model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model    

def train_model(X, y, window_size, step_size, m):
    # Split the data into sliding windows
    windows_X, windows_y = sliding_window(X, y, window_size, step_size)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(windows_X, windows_y, test_size=0.2, random_state=42)
    
    if m == "RF":
        model = rF(X_train, y_train)
    elif m == "SVC":
        model = sVC(X_train, y_train)
    elif m == "KN":
        model = kN(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    
    # Calculate the accuracy of the model
    acc = accuracy_score(y_test, y_pred)
    
    # Calculate F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')
    return acc, f1, model

def test_model(model, X_new, y_new, window_size, step_size):
    # Split the data into sliding windows
    windows_X_new, windows_y_new = sliding_window(X_new, y_new, window_size, step_size)

    # Make predictions on the testing data
    y_pred_new = model.predict(windows_X_new.reshape(windows_X_new.shape[0], -1))
    
    # Calculate the accuracy of the model
    acc_new = accuracy_score(windows_y_new, y_pred_new)
    
    print(y_pred_new)
    
    # Calculate F1-score
    f1 = f1_score(windows_y_new, y_pred_new, average='weighted')
    
    return acc_new, f1

# Evaluate the performance of the classification models
def evaluate_classification(X, y, testdf, models, window_size, step_size):



    # Initialize the results dictionary
    results = {}

    # Loop through the models
    for m in models:
        print("Training model: ", m)

        # Train the model on the training data
        acc_train, f1_train,model = train_model(X, y, window_size, step_size, m)

        # Test the model on the testing data
        X_test, y_test = clean_data(testdf, datacolumns)
        
        acc_test, f1_test = test_model(model, X_test, y_test, window_size, step_size)
       
        # Print the results
        print("Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(acc_train * 100, acc_test * 100))
        print(f"Train F1-score: {f1_train:.2f}, Test F1-score: {f1_test:.2f}")
        print()
        
models = ["RF", "SVC", "KN"]

# Clean the data and normalize
X, y = clean_data(df, datacolumns)

# Set the interval for finding local minima
interval = 140
windows = local_minima(X, interval)

window_size = windows
step_size = int(window_size)

results = evaluate_classification(X, y, testdf, models, window_size, step_size)

