import LoadData

from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import sys

np.random.seed(123)

############################################################################ KNN

def euclidean_distance(row1, row2):
    distance = 0.0
    i = 1
    while i in range(len(row1)):
        #print(row1)
        distance += (row1[i] - row2[i])**2
        i = i+1
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def get_neighborsIndexes(train, test_row, num_neighbors):
    distances = list()

    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    #print(distances)
    indexes = list()

    for i in range(num_neighbors):
        min = sys.maxsize
        minI = 0
        for j in range(len(distances)):
            if distances[j][1]< min and j not in indexes:
                min = distances[j][1]
                minI = j
        indexes.append(minI)

    return indexes

#predicts a classification for a dataset
def predict_classification(data,labels, row, num_neighbors):

    #neighbors = get_neighbors(data, row, num_neighbors)
    neighbors = get_neighborsIndexes(data, row, num_neighbors)

    #print(neighbors)

    output_values = list()
    for ind in neighbors:
        output_values.append(labels[ind])

    #output_values = [row for row in neighbors] #index from neighborIndexes
    #print(output_values)
    prediction = max(set(output_values), key=output_values.count)
    #print(prediction)
    return prediction

#find restults for all the
def k_nearest_neighbors(data,dataLabels, test, k):
    res_list = []
    i = 1

    dataNp = data.to_numpy()
    testNp = test.to_numpy()

    for row in testNp:
        output = predict_classification(dataNp,dataLabels, row, k)
        res_list.append(output)
        #break
        #print("row completed"+str(i))
        i = i+1

    return(res_list)

#builds a full confusion matrix for the dataset
def build_confusion_matrix(trueVals, predictions, k):

    #initialize a 10x10 confusion matrix for all digit representations. real values on vert. predicted on horizontal
    matrix = np.zeros([k,k])

    for i in range(len(predictions)):
        true_val = int(trueVals[i])
        pred_val = int(predictions[i])
        matrix[true_val][pred_val] = matrix[true_val][pred_val] + 1
        #print(str(true_val)+" "+str(pred_val)+" "+str(matrix[true_val][pred_val]))


    labels = np.zeros(k)
    for i in range(len(labels)):
        labels[i] = i

    print("i|"+str(labels))
    for i in range(len(matrix)):
        print(str(i)+"|"+str(matrix[i]))

    return

#calculates the accuracy of a set of predictions when clusters = 2
def binaryAccuracy(trueVals,predictions):

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(predictions)):
        true_val = trueVals[i]
        pred_val = predictions[i]
        if true_val == pred_val and pred_val == 1:
            true_pos = true_pos+1
        elif true_val == pred_val and pred_val == 0:
            true_neg = true_neg+1
        elif true_val != pred_val and pred_val == 1:
            false_pos = false_pos+1
        elif true_val != pred_val and pred_val == 0:
            false_neg = false_neg+1
        else:
            print(str(true_val)+" and "+str(pred_val))

    accuracy = float(true_pos+true_neg)/(true_pos+true_neg+false_neg+false_pos)

    #print statements
    #print(accuracy)
    print('accuracy: ', (accuracy))
    print('True Positive: %d' % (true_pos))
    print('True Negative: %d' % (true_neg))
    print('False Positive: %d' % (false_pos))
    print('False Negative: %d' % (false_neg))

    return

def KNN_suggest_nation(data,test, k, ideal, current):
    #TODO
    return

#################################################################################################### features

#drops unneeded features from the dataframe and stores them in another set
def featureEngineering(x):

    #drop score from dataset
    Data = x.drop(['Score'], axis=1)

    #store ranks and nations
    Zed = Data[['Overall rank','Country or region']]
    Data = Data.drop(['Country or region', 'Overall rank'], axis=1)

    #print(Data)
    #print(Zed)

    return Data,Zed

#given a set of scores, splits the data into brackets to justify groups of similar hapinness scores
def bracketScores(y, numbrackets):

    buckets = np.zeros(y.shape)

    mx = np.max(y)
    mn = np.min(y)

    length_of_range = (mx - mn + 1) / numbrackets

    rangeArray = np.zeros([numbrackets,2])
    start_of_range = mn
    end_of_range = start_of_range + length_of_range
    i=0
    rangeArray[i] = [start_of_range,end_of_range]
    while i<numbrackets-1:
        start_of_range = end_of_range
        end_of_range = start_of_range + length_of_range
        i = i+1
        rangeArray[i] = [start_of_range, end_of_range]

    #print(mn)
    #print(mx)
    #print(rangeArray)

    for i in range(len(y)):
        val = y[i]
        brac = 0
        for j in range(len(rangeArray)):
            if val>= rangeArray[j][0] and val<rangeArray[j][1]:
                brac = j
                break
        buckets[i] = brac

    #print(y)
    #print(buckets)

    return buckets

if __name__ == '__main__':
    x2019, y2019 = LoadData.load2019()       #use 2019 as test data
    x2019,z2019 = featureEngineering(x2019)

    x2018, y2018 = LoadData.load2018()          #use2018 as train data
    x2018, z2018 = featureEngineering(x2018)

    yBracket19 = bracketScores(y2019, 4)
    yBracket18 = bracketScores(y2019, 4)

    #KNN clutering with K = 5v
    print("\nresults for clustering with k=5 and n=4")
    y2019res = k_nearest_neighbors(x2018,yBracket18, x2019,5)
    #print(y2019res)

    build_confusion_matrix(yBracket19,y2019res,4)

    #KNN clustering with K = 3
    print("\nresults for clustering with k=3 and n=4")
    y2019res = k_nearest_neighbors(x2018, yBracket18, x2019, 3)
    #print(y2019res)

    build_confusion_matrix(yBracket19, y2019res, 4)

    # KNN clustering with K = 2
    print("\nresults for clustering with k=3 and n=2")
    yBracket19 = bracketScores(y2019, 2)
    yBracket18 = bracketScores(y2019, 2)

    y2019res = k_nearest_neighbors(x2018, yBracket18, x2019, 3)
    #print(y2019res)

    build_confusion_matrix(yBracket19, y2019res, 2)
    binaryAccuracy(yBracket19,y2019res)

    ##################################################### prediction section based on KNN