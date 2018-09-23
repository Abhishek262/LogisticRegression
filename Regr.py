# -*- coding: cp1252 -*-
import numpy as np
import csv,random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class LogisticRegression():
    def __init__(self, X, y, learning_rate=0.5, number_of_iterations=2000, plot=True):
        #Holds the coefficient values
        self.__W = [1,1,1]
        self.__b = 0.
        #Holds values of x1,x2,x3 in this case
        self.__X = X
        self.__y = y
        self.__m = len(X)
        self.__learning_rate = learning_rate
        self.__number_of_iterations = number_of_iterations
        self.__grads = {}
        self.__plot = plot


    #Returns the value of H(theta) for a particular x1,x2,x3 in X
    def HT(self,rowX) :
        ht = 0
        for j in range(0,len(self.__X[0])) :
            ht += rowX[j]*self.__W[j]
            
        
        return sigmoid(ht+ self.__b)


    #cost for a particular row
    def __compute_cost(self):
        cost = 0
        for i in range(0,self.__m) :
            cost += self.cfunc(self.HT(self.__X[i]),self.__y[i])

        return cost
            

    def cfunc(self,ht,y) :
        return (-y*np.log(ht) - (1-y)*np.log(1-ht))

    #gives derevative of cost func for every column in X
    def __cost_der(self,col) :
        costder = 0
        for i in range(self.__m ) :
            costder += ((self.HT(self.__X[i]) - self.__y[i]))*self.__X[i][col]
            
        return costder
        
    
    #Runs GD to update on __W values    
    def GradientDescent(self) :
        ctT  = 0
        coeff = []
        costs = []

        for theta in self.__W :

            for iters  in range(self.__number_of_iterations) :
                r = self.__learning_rate * float((1.0/float(self.__m))) * self.__cost_der(ctT)
                theta= theta  -r
                self.__W[ctT] = theta
                
                if iters%100 ==0 :
                    templ = []
                    print("value of theta",ctT+1," at iteration",iters,"is",theta[0])
                    cost = self.__compute_cost()
                    costs.append(cost)

            coeff.append(theta)
            ctT +=1
            print()
        #Plots the graph of cost vs no. of iterations(per 100) at a particular learning rate. 
            if self.__plot == True:
                plt.plot(np.squeeze(costs))
                plt.ylabel('Cost')
                plt.xlabel('Number of iterations (per hundreds)')
                plt.title("Learning rate =" + str(self.__learning_rate))
                plt.show()
         

    
    def predict(self, X, y=None):
        predicted = np.zeros((1, X.shape[1]))
        activation = self.__activate(X)
        for i in range(activation.shape[1]):
            predicted[0, i] = (activation[0, i] > 0.5)
        if y != 'empty':
            self.__confusion_matrix = confusion_matrix(y.T, predicted.T)
        return predicted
    
    def get_weights(self):
        return self.__W

    def get_bias(self):
        return self.__b

    def get_confusion_matrix(self):
        return self.__confusion_matrix


#Function to read stuff because pandas gives too many warnings for some reason
def read(foo):

    fil = csv.reader(foo,  delimiter = ',')
    line_count = 0
    data = []
    Traindata = []
    Testdata = []
    X = []
    Y = []
    
    for row in fil :
        if line_count == 0 :
            line_count +=1
            continue
        else :
            temp = []
            tempX = []
            tempY = []

            temp.append(float(row[0]))
            temp.append(float(row[1]))
            temp.append(float(row[2]))
            temp.append(float(row[3]))

        
            line_count +=1
            data.append(temp)
            tempX.append(float(row[0]))
            tempX.append(float(row[1]))
            tempX.append(float(row[2]))
            X.append(tempX)
            tempY.append(float(row[3]))
            Y.append(tempY)

    line_count = line_count -1
    start = random.randrange(10,line_count/2)
    end=  random.randrange(line_count/2,line_count)
    for i  in range(start,end) :
        Traindata.append(data[i])
    
    for i in range(0,start) :
        Testdata.append(data[i])
    for i in range(end,line_count) :
        Testdata.append(data[i])
    

    return Traindata, Testdata , data , np.array(X) , np.array(Y)

def L1_loss(y_hat, y):
    return np.sum(np.abs(y_hat - y))

def L2_loss(y_hat, y):
    return np.sum((y_hat - y)**2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Normalization function
def normalizeRows(X):
    x_forb_norm = np.linalg.norm(X, ord = None, axis = 1, keepdims = True)
    return X/ x_forb_norm


#All the data reading stuff
rawDatadir = file('Dataset\Haberman.csv','rb')
trainData,testData,fulldata ,matX,matY= read(rawDatadir)
matX = normalizeRows(matX)

#make a Logistic regressor object for Haberman's dataset for training data
trainset = LogisticRegression(matX,matY,0.03,1400)
coeff = hb.GradientDescent()
print()

print(coeff)
