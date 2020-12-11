import LoadData
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math

def happinessRegression_GDP(x2018,y2018,x2019,y2019):

    Labels18 = x2018['GDP per capita']
    Labels19 = x2019['GDP per capita']
    train = np.array(Labels18)
    train = np.reshape(train,(len(train), 1))
    test = np.array(Labels19)
    test = np.reshape(train,(len(test), 1))

    #print(train)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(train, y2018)

    # Make predictions using the testing set
    preds = regr.predict(test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y2019, preds))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y2019, preds))

    # Plot outputs
    plt.scatter(test, y2019, color='black')
    plt.plot(test, preds, color='blue', linewidth=3)
    plt.title("Regression of GDP to predict happiness")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")
    plt.yscale
    plt.xscale

    plt.xticks(())
    plt.yticks(())

    plt.show()

    return

def happinessRegression_SoS(x2018,y2018,x2019,y2019):

    Labels18 = x2018['Social support']
    Labels19 = x2019['Social support']
    train = np.array(Labels18)
    train = np.reshape(train,(len(train), 1))
    test = np.array(Labels19)
    test = np.reshape(train,(len(test), 1))

    #print(train)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(train, y2018)

    # Make predictions using the testing set
    preds = regr.predict(test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y2019, preds))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y2019, preds))

    # Plot outputs
    plt.scatter(test, y2019, color='black')
    plt.plot(test, preds, color='blue', linewidth=3)
    plt.title("Regression of Social support to predict happiness")
    plt.xlabel("Social support score")
    plt.ylabel("Happiness Score")


    plt.xticks(())
    plt.yticks(())

    plt.show()

    return

def happinessRegression_FullSet(x2018,y2018,x2019,y2019):

    Labels18 = x2018.drop(['Score','Overall rank','Country or region'], axis=1)
    Labels19 = x2019.drop(['Score','Overall rank','Country or region'], axis=1)

    train = np.array(Labels18)
    test = np.array(Labels19)

    deleteRowTrain = list()
    deleteRowTest = list()

    #remove nan rows
    for i in range(len(train)):
        for x in train[i]:
            if math.isnan(x):
                deleteRowTrain.append(i)

    # remove nan rows in test
    for i in range(len(test)):
        for x in test[i]:
            if math.isnan(x):
                deleteRowTest.append(i)

    for i in deleteRowTrain:
        train = np.delete(train,i,0)
        y2018 = np.delete(y2018,i,0)

    for i in deleteRowTest:
        test = np.delete(test,i,0)
        y2019 = np.delete(y2019,i,0)

    #print(train)

    # Create linear regression object
    regr = linear_model.LinearRegression()


    # Train the model using the training sets
    regr.fit(train, y2018)

    # Make predictions using the testing set
    preds = regr.predict(test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y2019, preds))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y2019, preds))



    return

def avg_timeRegression_GDP(x2015,y2015,x2016,y2016,x2017,y2017,x2018,y2018,x2019,y2019):

    train = np.zeros((4,2))
    test = np.zeros((1,2))
    results = np.zeros(4)

    print(train)

    Labels15 = x2015['Economy (GDP per Capita)']
    Labels16 = x2016['Economy (GDP per Capita)']
    Labels17 = x2017['Economy..GDP.per.Capita.']
    Labels18 = x2018['GDP per capita']
    Labels19 = x2019['GDP per capita']

    train[0] = [1,Labels15.mean()]
    train[1] = [2,Labels16.mean()]
    train[2] = [3,Labels17.mean()]
    train[3] = [4,Labels18.mean()]
    test[0] = [5,Labels19.mean()]

    results[0] = y2015.mean()
    results[1] = y2016.mean()
    results[2] = y2017.mean()
    results[3] = y2018.mean()

    testR = list()
    testR.append(y2019.mean())

    #train = np.reshape(train, (len(train), 1))

    #print(train)
    #print(results)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(train, results)

    #predict 2019
    preds = regr.predict(test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Predicted result for 2019"+str(preds))
    print("actual result for 2019" + str(testR))


    return


if __name__ == '__main__':

    x2019, y2019 = LoadData.load2019()
    x2018, y2018 = LoadData.load2018()
    x2017, y2017 = LoadData.load2017()
    x2016, y2016 = LoadData.load2016()
    x2015, y2015 = LoadData.load2015()


    print("\n Regression results for GDP vs happiness. 2018 train/2019 test")
    #happinessRegression_GDP(x2018,y2018,x2019,y2019)

    print("\n Regression results for Social support vs happiness. 2018 train/2019 test")
    happinessRegression_SoS(x2018,y2018,x2019,y2019)

    print("\n MultiRegression results for features vs happiness. 2018 train/2019 test")
    #happinessRegression_FullSet(x2018,y2018,x2019,y2019)


    print("\n Regression results for GDP vs happiness over time. avg of 2015-2018 train")
    #avg_timeRegression_GDP(x2015,y2015,x2016,y2016,x2017,y2017,x2018,y2018,x2019,y2019)