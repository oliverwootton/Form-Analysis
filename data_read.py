import csv
import pandas as pd
            
def yData(filename):
    data = []
    df = pd.read_csv(filename)
    for row in df.iterrows():
        data.append(row[1]["DMUAccelY"])
    return data

def xData(filename):
    data = []
    df = pd.read_csv(filename)
    for row in df.iterrows():
        data.append(row[1]["DMUAccelX"])
    return data

def recNo(filename):
    data = []
    df = pd.read_csv(filename)
    for row in df.iterrows():
        data.append(row[1]["RecNo"])
    return data