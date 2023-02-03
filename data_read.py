import csv
import pandas as pd

class Data:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
  
    def xData(self):
        return [row[1]["DMUAccelX"] for row in (self.df).iterrows()]
                
    def yData(self):
        return [row[1]["DMUAccelY"] for row in (self.df).iterrows()]

    def zData(self):
        return [row[1]["DMUAccelZ"] for row in (self.df).iterrows()]

    def recNo(self):
        return [row[1]["RecNo"] for row in (self.df).iterrows()]
