from math import sqrt
import csv
import time

#reads in data into a matrix of doubles from the .csv
def read_in_data():
    data = []

    with open('diabetes.csv', newline='') as file:
        reader = csv.reader(file)
        header = True
        for row in reader:
            if header == False:
                dataRow = []
                for i in range(len(row)):
                    dataRow.append(float(row[i]))
                data.append(dataRow)
            else:
                header = False;
            #print(row)
    return data

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# calculate the manhatten distance between two vectors
def manhatten_distance(row1, row2):
    distance = 0;
    for i in range(len(row1)-1):
        distance += abs(row1[i] - row2[i])
    return distance

# calculate the minikowski distance with a p=3
def minikowski_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += abs(row1[i] - row2[i])**3
    return distance**(1/3)

#gets the nearest neighbors for a given row in the data set
def get_neighbors(data, dataRow, neighborNum, mode):
    distances = []
    for train_row in data:

        if mode == 1:
            dist = euclidean_distance(dataRow, train_row)
        elif mode ==2:
            dist = manhatten_distance(dataRow, train_row)
        else:
            dist = minikowski_distance(dataRow, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(neighborNum):
        neighbors.append(distances[i][0])

    return neighbors

#predicts a value for a row using the KNN for a
def predict_classification(data, dataRow, neighborNum, mode): # change to calculate the mean of the output values
    neighbors = get_neighbors(data, dataRow, neighborNum, mode)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)

    return prediction

#run KNN for all rows in all distance methods
def k_nearest_neighbors(test):
    res_list = []
    predictions1 = []
    predictions2 = []
    predictions3 = []
    for row in test:
        output1 = predict_classification(test, row, 3, 1)
        output2 = predict_classification(test, row, 3, 2)
        output3 = predict_classification(test, row, 3, 3)
        predictions1.append(output1)
        predictions2.append(output2)
        predictions3.append(output3)
    res_list.append(predictions1)
    res_list.append(predictions2)
    res_list.append(predictions3)
    return(res_list)

#calculates the accuracy of a set of predictions
def analyze(data,predictions):

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(predictions)):
        true_val = data[i][-1]
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

if __name__ == '__main__':
    data = read_in_data()
    print(data[73]) #make sure data is read in correctly

    #make a prediction for a specific row
    prediction = predict_classification(data, data[73], 3, 1)
    print('Expected %d, Got %d.' % (data[73][-1], prediction))

    t0 = time.clock()
    #find result list for all distance methods
    res_list = k_nearest_neighbors(data)
    t1 = time.clock()

    print("Results for Euclidean distance")
    analyze(data,res_list[0])
    print("Results for Manhatten Distance")
    analyze(data,res_list[1])
    print("Results for Minikowski Distance")
    analyze(data,res_list[2])

    completion_time = t1-t0
    print("time to complete all K-NN algorithms/predictions(s): "+str(completion_time))

