from math import sqrt
import time
import csv

#reads in data into a matrix of doubles from the .csv
def read_in_data():
    data = []
    test = []

    with open('train.csv', newline='') as file:
        reader = csv.reader(file)
        header = True
        counter = 0
        for row in reader:
            if header == False:
                dataRow = []
                for i in range(len(row)):
                    dataRow.append(float(row[i]))
                if counter < 50:
                    data.append(dataRow)
                elif counter >= 50 and counter <100:
                    test.append(dataRow)
                elif counter >= 100:
                    return data,test
                counter = counter+1
            else:
                header = False;
            #print(row)
    return data,test

# calculate the Euclidean distance between two vectors (label is in spot 0)
def euclidean_distance(row1, row2):
    distance = 0.0
    i = 1
    while i in range(len(row1)):
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

#predicts a classification for a dataset
def predict_classification(data, row, num_neighbors):
    neighbors = get_neighbors(data, row, num_neighbors)
    output_values = [row[0] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

#find restults for all the
def k_nearest_neighbors(data,test, k):
    res_list = []
    i = 1
    for row in test:
        output = predict_classification(data, row, k)
        res_list.append(output)
        #print("row completed"+str(i))
        i = i+1

    return(res_list)

#builds a full confusion matrix for the dataset
def build_confusion_matrix(data, predictions):

    #initialize a 10x10 confusion matrix for all digit representations. real values on vert. predicted on horizontal
    matrix = []
    for i in range(10):
        matrix.append([0,0,0,0,0,0,0,0,0,0])

    for i in range(len(predictions)):
        true_val = int(data[i][0])
        pred_val = int(predictions[i])
        matrix[true_val][pred_val] = matrix[true_val][pred_val] + 1
        #print(str(true_val)+" "+str(pred_val)+" "+str(matrix[true_val][pred_val]))


    labels = [0,1,2,3,4,5,6,7,8,9]
    print("i|"+str(labels))
    for i in range(len(matrix)):
        print(str(i)+"|"+str(matrix[i]))

    #print("untested")

def accuracy_for_digit_9(data, predictions):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(predictions)):
        true_val = data[i][0]
        pred_val = predictions[i]
        if true_val == pred_val and true_val == 9:
            true_pos = true_pos + 1
        elif true_val == pred_val and true_val != 9:
            true_neg = true_neg + 1
        elif true_val != 7 and pred_val == 9:
            false_pos = false_pos + 1
        elif pred_val != 7 and true_val == 9:
            false_neg = false_neg + 1
        #else:
         #   print(str(true_val) + " and " + str(pred_val))

    accuracy = float(true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos)

    # print statements
    # print(accuracy)
    print('accuracy: ', (accuracy))
    print('True Positive: %d' % (true_pos))
    print('True Negative: %d' % (true_neg))
    print('False Positive: %d' % (false_pos))
    print('False Negative: %d' % (false_neg))

if __name__ == '__main__':
    print("Read Data")
    data,test = read_in_data()
    print(data[6])
    print(test[3])

    # make a prediction for a specific row

    prediction = predict_classification(data, test[3], 3)
    print('Expected %d, Got %d.' % (test[3][0], prediction))

    print("Start KNN")
    t0 = time.clock()
    # find result list for all distance methods
    res_list = k_nearest_neighbors(data,test, 3)
    t1 = time.clock()

    build_confusion_matrix(test,res_list)
    accuracy_for_digit_9(test, res_list)
    completion_time = t1 - t0
    print("time to complete K-NN algorithms/predictions(s): " + str(completion_time))


    res_list2 = k_nearest_neighbors(data,test, 1)
    res_list3 = k_nearest_neighbors(data,test, 10)

    print("how accuracy varies with k = 1")
    accuracy_for_digit_9(data, res_list2)
    print("how accuracy varies with k=10")
    accuracy_for_digit_9(data, res_list3)