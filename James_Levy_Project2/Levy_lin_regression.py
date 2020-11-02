from helperfunctions import *
from scipy.io import loadmat
from numpy.matlib import repmat
import numpy as np
import matplotlib.pyplot as plt
import csv

#reads in data
def read_in_data():
    data = []
    test = []

    with open('kc_house_data.csv', newline='') as file:
        reader = csv.reader(file)
        header = True
        counter = 0
        for row in reader:
            if header == False:
                dataRow = []
                for i in range(len(row)):
                    dataRow.append(row[i])
                if counter < 1000:
                    data.append(dataRow)
                elif counter >= 1000 and counter <2000:
                    test.append(dataRow)
                elif counter >= 2000:
                    return data,test
                counter = counter+1
            else:
                header = False;
            #print(row)
    return data,test

#takes a dataset and splits it into price and space vetors
def XYlists(dataSet):

    #print(len(dataSet))
    x = np.empty([len(dataSet)])
    y = np.empty([len(dataSet)])

    for i in range(len(dataSet)):
        x_var = float(dataSet[i][5]) #sqrt living
        y_var = float(dataSet[i][2]) #price
        #print(x_var)
        x[i] = x_var
        y[i] = y_var

    return x,y

#cost functions
def cost_f(y_current, yList, N):
    return sum([element ** 2 for element in (yList - y_current)]) / N

#this version of the function causes overflow errors for larger interations/epoch
def gradientDecent(xList, yList, iter):
    N = float(len(yList))
    learning_rate = .0001
    m = 0
    b = 0

    for i in range(iter): #number of iterations for descent

        Y_pred = m * xList + b  # The current predicted value of Y

        cost = cost_f(Y_pred, yList, N)

        D_m = (-2 / N) * sum(xList * (yList - Y_pred))  # Derivative wrt m
        D_c = (-2 / N) * sum(yList - Y_pred)  # Derivative wrt c
        m = m - learning_rate * D_m  # Update m
        b = b - learning_rate * D_c  # Update c

    return m, b, cost


# finds a sample coeficient without using cost functions and gradient decent
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)

# used for plotting the line
def plot_regression_line(x, y, m,b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b + m * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


def analyzeAccuracy(xTest,yTest,b):
    print("accuracy")
    sum = 0
    for i in range(len(xTest)):
        y_pred = b[0] + b[1] * xTest[i]
        y_actual = yTest[i]
        sum = sum + (y_pred-y_actual)

    avg_dist = sum/len(xTest)
    print("On Avg, predicted value is this distance away from the actual value", avg_dist)


if __name__ == '__main__':
    print("start to linear regression")  #square foot living to predict price

    #xTr, yTr, xTe, yTe = loaddata("kc_house_data.csv") #giving error. email professor

    print("Read Data")
    data, test = read_in_data()

    xTrain,yTrain = XYlists(data)
    xTest, yTest = XYlists(test)

    print("\n~~~~~~~~~~~~~~RESULTS FOR SIMPLE LIN_REG~~~~~~~~~~~~~~~\n")

    b = estimate_coef(xTrain, yTrain)
    print("Estimated coefficients:\nb = {} m = {}".format(b[0], b[1]))

    print("example prediction")
    print("price for house with this square footage: ", xTest[10])
    y_pred = b[0] + b[1] * xTest[10]
    y_actual = yTest[10]
    print("price estimate: ", y_pred)
    print("price actual", y_actual)

    analyzeAccuracy(xTest, yTest, b)

    plot_regression_line(xTrain, yTrain, b[1], b[0])

    print("\nRESULTS FOR GRADIENT LIN_REG ~~~ seems to overflow before convergence for large epoch/iterations\n")

    b2 = gradientDecent(xTrain,yTrain,1)
    print("gradient decent coefficients when run once(m, b, cost): ",b2)

    print("example prediction")
    print("price for house with this square footage: ", xTest[10])
    y_pred = b2[1] + b2[0] * xTest[10]
    y_actual = yTest[10]
    print("price estimate: ", y_pred)
    print("price actual", y_actual)

    plot_regression_line(xTrain, yTrain, b2[0], b2[1])


