import pandas as pd
from helper_functions import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score 
from sklearn.model_selection import train_test_split

columns = ["TimeStamp", "MoveType", "AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]
datacolumns = ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]

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
    X_train, X_test, y_train, y_test = train_test_split(windows_X, windows_y, test_size=0.5, random_state=42)
    
    # Train the specific model
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
    
    return acc, model

def test_model(model, X_new, y_new, window_size, step_size):
    # Split the data into sliding windows
    windows_X_new, windows_y_new = sliding_window(X_new, y_new, window_size, step_size)

    # Make predictions on the testing data
    y_pred_new = model.predict(windows_X_new.reshape(windows_X_new.shape[0], -1))

    return y_pred_new

def analyseForm(df, squatData, m):
    # Clean the data and normalize
    X, y = clean_data(df, datacolumns)
    X_test, y_test = clean_data(squatData, datacolumns)
    
    # Set the interval for finding local minima
    interval = 100
    windows, minima, local_minima = local_Minima(X_test, interval)
    window_size = windows
    step_size = windows
    print("Window size: " + str(window_size))
    print("Step size: " + str(step_size))
    
    # Train the model on the training data
    acc_train, model = train_model(X, y, window_size, step_size, m)

    # Analyse the form of new input data
    y_prediction = test_model(model, X_test, y_test, window_size, step_size)
    
    return y_prediction, len(minima), window_size, step_size

# Dataset to train the model
training_file = "AllData"
# Data to be analysed
new_data = "testAll"

# Read the data into data frames
df = pd.read_csv(training_file + ".csv")
data_to_analyse = pd.read_csv(new_data + ".csv")

# Remove any columns that are unnecessary 
df = df[columns]
data_to_analyse = data_to_analyse[columns]

# Machine learning model used
m = ("RF")

# Analyse the form of the new data
results, reps, window_size, step_size = analyseForm(df, data_to_analyse, m)

# Print statements of the results produced by the model
print("The number of reps completed: " + str(reps))
print("The window size used: " + str(window_size))
print("The step size used: " + str(step_size))

print("The flowing is a break done of the technique identified on each repetition:")
i = 1
for rep in results:
    print("Rep " + str(i))
    if rep == "misc1":
        print("This rep used the correct form\n")
    elif rep == "misc2":
        print("This rep lacked depth\n")
    elif rep == "misc3":
        print("This rep pushed through your toes instead of heels\n")
    elif rep == "misc4":
        print("This rep had too much bend in your back\n")     
    i += 1
    
