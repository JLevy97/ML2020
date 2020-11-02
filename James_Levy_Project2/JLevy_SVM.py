import numpy as np
import csv
import time

lr = .001 #a random learning rate for our SVM
lambda_param = .01

#read in data in a customized way
def read_in_data():
    data = []
    dataSoltions = []
    test = []
    testSolutions = []

    with open('breast-cancer-wisconsin.data', newline='') as file:
        reader = csv.reader(file)
        counter = 0
        for row in reader:

            if '?' not in row:
                dataRow = []

                for i in range(len(row)-2):
                    dataRow.append(float(row[i+1]))
                #print(dataRow)

                if counter%4 != 0:
                    data.append(dataRow)
                    dataSoltions.append(float(row[-1]))
                else:
                    test.append(dataRow)
                    testSolutions.append(float(row[-1]))

                counter = counter+1

            #print(row)
    return data,test,dataSoltions,testSolutions

def train(x, y):
    n_samples, n_features = x.shape

    y_ = np.where(y <= 2, -1, 1) #sets it up where labels of 2 are correctly denoted as negative and labels of 4 are denoted as positve

    #initial weights for training
    w = np.zeros(n_features)
    b = 0

    #gradient_decent loop
    for iter in range(1000): #max number of iterations/ epochs for the maximization algo
        for idx, row in enumerate(x): #loop to update weights based on the cost
            condition = y_[idx] * (np.dot(row, w) - b) >= 1  # calculate the condition to check for updating weights
            if condition: #update using derivatives of needed cost functions
                w -= lr * (2 * lambda_param * w)
            else:
                w -= lr * (2 * lambda_param * w - np.dot(row, y_[idx])) #update the weights
                b -= lr * y_[idx]

    return w,b


#predicts a single row using linear model
def predict(x, w, b):
    guess = np.dot(x, w) - b
    estimate =  np.sign(guess)

    #takes classification sign and switches it back to original labels.
    if estimate > 0:
        return 4.0
    else:
        return 2.0

#analyze results for the entire test set
def analyze(testSet, testSolutions, w, b):
    res = []
    for row in testSet:
        res.append(predict(row,w,b))

    #print(res)
    res = np.array(res)
    #res = np.where(res <= 0, 2.0, 4.0)
    #print(res)

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(testSolutions)):
        true_val = res[i]
        pred_val = testSolutions[i]
        if true_val == pred_val and pred_val == 4:
            true_pos = true_pos+1
        elif true_val == pred_val and pred_val == 2:
            true_neg = true_neg+1
        elif true_val != pred_val and pred_val == 4:
            false_pos = false_pos+1
        elif true_val != pred_val and pred_val == 2:
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

if __name__ == '__main__':
    print("Read Data")
    data, test, dataSol, testSol = read_in_data()

    data = np.array(data)
    test = np.array(test)
    dataSol = np.array(dataSol)
    testSol = np.array(testSol)

    t0 = time.clock()
    w,b = train(data,dataSol)
    t1 = time.clock()

    print("results of training the SVM")
    print("weights: ", w)
    print("bias-value: ",b)
    print()

    #print(predict(test[7],w,b))
    #print(testSol[7])

    print("time to create SVM hyperplane fit: ",(t1-t0))
    analyze(test,testSol,w,b)