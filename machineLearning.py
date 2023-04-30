import pandas as pd
from helper_functions import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score 
from sklearn.model_selection import train_test_split

columns = ["TimeStamp", "MoveType", "AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]
datacolumns = ["AccelroX", "AceelroY", "AceelroZ", "DMPitch", "DMRoll", "DMYaw", "DMGrvX", "DMGrvY", "DMGrvZ"]

models = ["RF", "SVC", "KNN"]

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

def kNN(X_train, y_train):
    # Create and fit the KNN model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model    

def train_model(X, y, window_size, step_size, m):
    # Split the data into sliding windows
    windows_X, windows_y = sliding_window(X, y, window_size, step_size)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(windows_X, windows_y, test_size=0.5, random_state=42)
    
    if m == "RF":
        model = rF(X_train, y_train)
    elif m == "SVC":
        model = sVC(X_train, y_train)
    elif m == "KNN":
        model = kNN(X_train, y_train)
    
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
    
    # Calculate F1-score
    f1 = f1_score(windows_y_new, y_pred_new, average='weighted')
    
    return acc_new, f1

# Evaluate the performance of the classification models
def evaluate_classification(df, testdf, models):
    # Clean the data and normalize
    X, y = clean_data(df, datacolumns)
    X_test, y_test = clean_data(testdf, datacolumns)
    
    # Set the interval for finding local minima
    interval = 100
    windows, minima, local_minima = local_Minima(X_test, interval)

    window_size = windows
    step_size = windows
    print("Window size: " + str(window_size))
    print("Step size: " + str(step_size) + "\n")

    # Loop through the models
    for m in models:
        print("Training model: ", m)

        # Train the model on the training data
        acc_train, f1_train,model = train_model(X, y, window_size, step_size, m)
        
        # Test the model on the testing data
        acc_test, f1_test = test_model(model, X_test, y_test, window_size, step_size)
       
        # Print the results
        print("Trained Data Accuracy: {:.2f}%, F1-score: {:.2f}".format(acc_train * 100, f1_train))
        print("Unseen Test Data Accuracy: {:.2f}%, F1-score: {:.2f}".format(acc_test * 100, f1_test))
        print() 

# Dataset to train the model
training_file = "AllData"
# Unseen test data (data not in the dataset file)
test_file = "test1"

# Read the data into data frames
df = pd.read_csv(training_file + ".csv")
testdf = pd.read_csv(test_file + ".csv")

# Remove any columns that are unnecessary 
df = df[columns]
testdf = testdf[columns]
  
# Evaluate the performance of each model   
results = evaluate_classification(df, testdf, models)
