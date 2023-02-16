import numpy as np

#Author: Ali Hassan Sharif
#Purpose: To demonstrate gradient descent on a linear regression which has 1 feature variable

#Our parameters, change any to what you like
#Keep the alpha variable between the range 0.0 - 1.0
#If the algorithm does not give expected results, reduce the alpha parameter, as taught by Andrew Ng :)

x_trainData = np.array([1.0, 1.5, 2.1, 2.5, 3.0, 3.4, 4.2, 5.0])   #features
y_trainData = np.array([300.0, 500.0, 650.0, 825.0, 1300, 1250, 1600, 2500])   #target value
initW = 0
initB = 0
iterations = 10000
alpha = 0.01

#This function is not needed for gradient descent, we will be using it's derrivative in the getGradients function, shown here if interested
def costFunction(xData, yData, w, b):
    m = xData.shape[0]
    cost = 0
    for i in range (m):
        cost += ((w*xData[i] + b) - yData[i])**2
    cost = cost / (2 * m)

#Calculates derrivatives of cost function for each of it's input variables
#In this case, our cost function has 2 inputs, w and b
def getGradients(xData, yData, w, b):
    m = xData.shape[0]
    tempW = 0
    tempB = 0
    for i in range (m):
        tempW += ((w*xData[i] + b) - yData[i]) * xData[i]
        tempB += ((w*xData[i] + b) - yData[i])
    tempW = tempW/m
    tempB = tempB/m
    return tempW, tempB

#Our gradient descent function
def gradientDescent(xData, yData, initW, initB, alpha, numIterations, gradFunc):
    #Not needed, but makes things readable
    w = initW;
    b = initB;
    # To check convergence, (I'm guessing) we need to take the difference between the previous iteration and the current iteration's gradients
    # If the difference between the gradients is miniscule, it means we have reached (or close to) an optimal w and b value 
    # Not sure if this approach is correct, not taught in the first module of Andrew NG's Supervised learning course
    prevWGrad = 0
    prevBGrad = 0
    differenceW = 10000
    differenceB = 10000
    #Here we go
    for i in range (numIterations):
        print(f"Iteration #{i}")
        gradW, gradB = gradFunc(xData, yData, w, b); #Get gradients for these w and b values
        if(i == 1): #Check if we have done 1 iteration so we can calculate convergence
            prevWGrad = gradW
            prevBGrad = gradB
        if(i > 1): #Get the difference between our current and previous gradients for convergence purposes
            differenceW = gradW - prevWGrad
            differenceB = gradB - prevBGrad
        if((differenceW < 0.0001 and differenceW > -0.0001) and (differenceB < 0.0001 and differenceB > -0.0001)): #Convergence reached (this is my own defintion of convergence)
            print("Reached convergence, halting calculations.")
            break
        prevWGrad = gradW;
        prevBGrad = gradB;
        w = w - alpha*gradW;
        b = b - alpha*gradB;
        print(f"gradient of w: {gradW}")
        print(f"gradient of b: {gradB}")
    return w, b;

w, b = gradientDescent(x_trainData, y_trainData, initW, initB, alpha, iterations, getGradients)

print(f"The best w and b for this data set is: {w}, {b}")
