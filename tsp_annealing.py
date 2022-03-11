# Referencia de: https://www.codestudyblog.com/cnb2105a/0515200057.html

import math                         #  math
import random                       #  random
import pandas as pd                 #  pandas  pd
import numpy as np                  #  numpy  np YouCans
import matplotlib.pyplot as plt     #  matplotlib.pyplot  plt

def getDistMat(nCities, coordinates):
    # Creating an empty array of size nCities x nCities
    distMat = np.zeros((nCities,nCities))

    for i in range(nCities):
        for j in range(i, nCities):
            # Getting the distance between the two coordinates
            distMat[i][j] = distMat[j][i] = round(np.linalg.norm(coordinates[i]-coordinates[j]))

    return distMat

#  TSP 
def calTourMileage(tourGiven, nCities, distMat):
    mileageTour = distMat[tourGiven[nCities-1], tourGiven[0]]

    for i in range(nCities-1):
        mileageTour += distMat[tourGiven[i], tourGiven[i+1]]

    return round(mileageTour)                     # 

#  PLOT TSP
def plot_tour(tour, value, coordinates):
    num = len(tour)
    x0, y0 = coordinates[tour[num - 1]]
    x1, y1 = coordinates[tour[0]]
    plt.scatter(int(x0), int(y0), s=15, c='r')      
    plt.plot([x1, x0], [y1, y0], c='b')             
    for i in range(num - 1):
        x0, y0 = coordinates[tour[i]]
        x1, y1 = coordinates[tour[i + 1]]
        plt.scatter(int(x0), int(y0), s=15, c='r') 
        plt.plot([x1, x0], [y1, y0], c='b')         

    plt.xlabel("Total mileage of the tour:{:.1f}".format(value))
    plt.title("Optimization tour of TSP{:d}".format(num))
    plt.show()

# Plot Optimization
def plot_Optimization(tourBest, valueBest, recordBest, recordNow, coordinates, nCities):
    plt.title("Optimization result of TSP{:d}".format(nCities)) # 
    plt.plot(np.array(recordBest),'b-', label='Best')           #  recordBest
    plt.plot(np.array(recordNow),'g-', label='Now')             #  recordNow
    plt.xlabel("iter")                                          #  x
    plt.ylabel("mileage of tour")                               #  y
    plt.legend()                                                # 
    plt.show()

    print("Tour verification successful!")
    print("Best tour: \n", tourBest)
    print("Best value: {:.1f}".format(valueBest))

def mutateSwap(tourGiven, nCities):
    # produce a mutation tour with 2-Swap operator
    # swap the position of two Cities in the given tour

    i = np.random.randint(nCities) 
    while True:
        j = np.random.randint(nCities)
        if i!=j: break 

    tourSwap = tourGiven.copy() 
    tourSwap[i],tourSwap[j] = tourGiven[j],tourGiven[i] #   i  j 

    return tourSwap

def main():
    tInitial = 100.0                # (initial temperature)
    tFinal  = 1                     # (stop temperature)
    nMarkov = 1000                  # Markov
    alfa    = 0.98                 # T(k)=alfa*T(k-1)

    # Opening txt file
    coordinates = []
    citiesFile = open("cities.txt", "r")
    for line in citiesFile:
        lineSplit = line.split()

        if (len(lineSplit) == 1): # Reading quantity of cities
            nCities = int(lineSplit[0])
        else: # Reading coordinates
            coordinates.append([int(lineSplit[0]), int(lineSplit[1])])

    # Create a np array and re-asigning the coordinates
    coordinates = np.array(coordinates)

    distMat = getDistMat(nCities, coordinates)  # Getting the distances between coordinates
    nMarkov = nCities                           # Markov 
    tNow    = tInitial                          #  (current temperature)

    tourNow   = np.arange(nCities)
    valueNow  = calTourMileage(tourNow,nCities,distMat) #  valueNow
    tourBest  = tourNow.copy()                          #  tourNow
    valueBest = valueNow                                #  valueNow
    recordBest = []                                     #  
    recordNow  = []                                     #  


    while tNow >= tFinal:
        for k in range(nMarkov):
            tourNew = mutateSwap(tourNow, nCities)      #   
            valueNew = calTourMileage(tourNew,nCities,distMat) # 
            deltaE = valueNew - valueNow

            #  Metropolis 
            if deltaE < 0:
                accept = True
                if valueNew < valueBest:
                    tourBest[:] = tourNew[:]
                    valueBest = valueNew
            else:
                pAccept = math.exp(-deltaE/tNow)
                if pAccept > random.random():
                    accept = True
                else:
                    accept = False

            if accept == True:
                tourNow[:] = tourNew[:]
                valueNow = valueNew

        tourNow = np.roll(tourNow,2)

        recordBest.append(valueBest)
        recordNow.append(valueNow)

        tNow = tNow * alfa

    # Plotting
    plot_tour(tourBest, valueBest, coordinates)
    plot_Optimization(tourBest, valueBest, recordBest, recordNow, coordinates, nCities)

main()