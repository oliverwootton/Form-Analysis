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
    """
    Trains a Random Forest classifier on the input data and returns the trained model.

    Parameters:
        X_train: An array (n_windows, window_size, n_features): The training input samples.
        y_train: An array (n_samples): The labels for the windows.

    Returns:
        model: The trained Random Forest classifier model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def sVC(X_train, y_train):
    """
    Trains a SVC model on the input data and returns the trained model.

    Parameters:
        X_train: An array (n_windows, window_size, n_features): The training input samples.
        y_train: An array (n_samples): The labels for the windows.

    Returns:
        model: The trained SVC model.
    """
    model = SVC(kernel='linear', C=1, gamma='auto', random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def kNN(X_train, y_train):
    """
    Trains a KNN model on the input data and returns the trained model.

    Parameters:
        X_train: An array (n_windows, window_size, n_features): The training input samples.
        y_train: An array (n_samples): The labels for the windows.

    Returns:
        model: The trained KNN model.
    """
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model   

def train_model(X, y, window_size, step_size, m):
    """
    Train a machine learning model using the given input data

    Parameters:
        X: The input features data
        y: The labels on for the feature data
        window_size: The size of the sliding window
        step_size: The step size for the sliding window
        m: The type of machine learning model to train. Must be one of 'RF', 'SVC', or 'KNN'.

    Returns:
        model: The trained machine learning model
    """
    # Split the data into sliding windows
    windows_X, windows_y = sliding_window(X, y, window_size, step_size)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(windows_X, windows_y, test_size=0.2, random_state=42)
    
    # Train the specific model
    if m == "RF":
        model = rF(X_train, y_train)
    elif m == "SVC":
        model = sVC(X_train, y_train)
    elif m == "KNN":
        model = kNN(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    
    return model

def test_model(model, X_new, y_new, window_size, step_size):
    """
    Test a machine learning model on new data using sliding windows.

    Parameters:
        model: A trained machine learning model.
        X_new: An array containing the unseen data's features.
        y_new: An array containing the unseen data's target labels.
        window_size: The integer of the sliding window.
        step_size: The integer for the step size of the sliding window.

    Returns:
        y_pred_new: An array containing the predicted labels for the new data.
    """
    # Split the data into sliding windows
    windows_X_new, windows_y_new = sliding_window(X_new, y_new, window_size, step_size)

    # Make predictions on the testing data
    y_pred_new = model.predict(windows_X_new.reshape(windows_X_new.shape[0], -1))

    return y_pred_new

def analyseForm(df, squatData, m):
    """
    Analyses the form of new input data using a machine learning model trained on existing data.

    Parameters:
        df: A pandas DataFrame containing the training data.
        squatData: A pandas DataFrame containing the unseen input data.
        m: A string specifying the machine learning model to use ("RF" for Random Forest, "SVC" for Support Vector 
           Classification, or "KNN" for k-Nearest Neighbors).

    Returns:
        A tuple containing:
        - An ndarray of predicted labels for the unseen input data.
        - The number of local minima found in the new input data.
        - The window size used for sliding window analysis.
        - The step size used for sliding window analysis.
    """
    # Clean the data and normalize
    X, y = preprocess_data(df, datacolumns)
    X_test, y_test = preprocess_data(squatData, datacolumns)
    
    # Set the interval for finding local minima
    interval = 100
    windows, minima, local_minima = local_Minima(X_test, interval)
    window_size = windows
    step_size = windows
    
    # Train the model on the training data
    model = train_model(X, y, window_size, step_size, m)

    # Analyse the form of new input data
    y_prediction = test_model(model, X_test, y_test, window_size, step_size)
    
    return y_prediction, len(minima), window_size, step_size

# Dataset to train the model
training_file = "Dataset/AllData"
# Data to be analysed
new_data = "UnseenData/NewTest copy"

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
    
