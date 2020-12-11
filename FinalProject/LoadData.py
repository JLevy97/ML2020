import numpy as np
import pandas as pd

def load2019():
    Data = pd.read_csv('2019.csv')
    print('Shape of the 2019 data set: ' + str(Data.shape))
    Labels = Data['Score']
    Labels = np.array(Labels)
    return Data, Labels

def load2018():
    Data = pd.read_csv('2018.csv')
    print('Shape of the 2018 data set: ' + str(Data.shape))
    Labels = Data['Score']
    Labels = np.array(Labels)
    return Data, Labels

    return

def load2017():

    Data = pd.read_csv('2017.csv')
    print('Shape of the 2017 data set: ' + str(Data.shape))
    Labels = Data['Happiness.Score']
    Labels = np.array(Labels)
    return Data, Labels

def load2016():

    Data = pd.read_csv('2016.csv')
    print('Shape of the 2016 data set: ' + str(Data.shape))
    Labels = Data['Happiness Score']
    Labels = np.array(Labels)
    return Data, Labels

def load2015():
    Data = pd.read_csv('2015.csv')
    print('Shape of the 2015 data set: ' + str(Data.shape))
    Labels = Data['Happiness Score']
    Labels = np.array(Labels)

    return Data, Labels