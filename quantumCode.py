from dimod import ConstrainedQuadraticModel, CQM, SampleSet
from dimod import Binary, quicksum
from dwave.system import LeapHybridCQMSampler
from dwave.cloud.client import Client
from dwave.cloud import config
import numpy as np
import pandas as pd
import ast
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
from scipy.spatial import distance
import time
import re

def genererateRandomCase(nbCities):
    """
    Function that generate a random case of CVRP and return the list of the cities and it's cost matrix
    """
    listCities = [[random.randint(-50,50),random.randint(-50,50)] for i in range(0, nbCities)]

    listCities.insert(0, [0,0])  #We add the repository in (0,0)
    nbCities+=1 #We add the repository

    #Generate Euclidean distances matrix
    costMatrix = generateCostMatrix(listCities)

    return listCities, costMatrix

def generateCostMatrix(listCities):
    """
    Function that generate the distance matrix of the input cities
    """
    costMatrix = []
    
    for i in range(0, len(listCities)):
        lineCostMatrix = []
        for j in range(0, len(listCities)):
            lineCostMatrix.append(distance.euclidean(listCities[i], listCities[j]))
        costMatrix.append(lineCostMatrix)
            
    return costMatrix

def Classification (nbOfPointToCluster, nbOfCluster, matrixOfCost, matrixOfCapacity, matrixOfVolume):
    """
    Return the clusturing of the input in the form of a dataframe : 
    "{'x0_0' : 0.0, 'x0_1' : 0.0, ..., x1_0 : 0.0, ..., x{nbOfPointToCluster}_{nbOfCluster}}", energy, is_feasible
 
    This function can be used to find the clustering of an actual problem.
 
    @type  nbOfPointToCluster: int

    @param nbOfPointToCluster: Number of point that will be subdivised in cluster

    @type  nbOfCluster: int
    @param nbOfCluster: Number of cluster that will subdivise our point

    @type  matrixOfCost: List of shape [nbOfPointToCluster, nbOfPointToCluster]
    @param matrixOfCost: Matrix that give the Cost between the point i and the point j,
                           j
                        | ... | i
                        | ... |
                        | ... |
    @type  matrixOfCapacity: List of shape [nbOfCluster]

    @param matrixOfCapacity: Matrix that give the Capacity of each cluster

    @type  matrixOfVolume: List of shape [nbOfPointToCluster]

    @param matrixOfVolume: Matrix that give the cost of each point of the city

    @rtype:   int

    @return:  the timer of the clustering.
    """
    #Define our model
    cqm=ConstrainedQuadraticModel()

   #Preparation of our matric that will got our solution
    x = {
    (i, d): Binary('x{}_{}'.format(i, d))
    for i in range(nbOfPointToCluster)
    for d in range(nbOfCluster)}

    #Objective function
    objective = quicksum(matrixOfCost[i][j] * x[(i,d)] * x[(j,d)]
        for i in range(nbOfPointToCluster)
        for j in range(i+1, nbOfPointToCluster)
        for d in range(nbOfCluster) )

    cqm.set_objective(objective)


    #Constraint
    #We want the repository in every cluster
    for d in range(nbOfCluster):
        cqm.add_constraint(x[(0,d)] == 1)

    #The sum of the capacity require by the point should not exceed the total capacity of the cluster
    for d in range(nbOfCluster):
        cqm.add_constraint(quicksum(matrixOfVolume[i] * x[(i,d)]
        for i in range(nbOfPointToCluster)) <= matrixOfCapacity[d])

    #Every point should be in 1 and only 1 cluster except the repository
    for i in range(1,nbOfPointToCluster):
        cqm.add_constraint(quicksum(x[(i,d)]
        for d in range(nbOfCluster)) == 1)


    #We get our solution
    cqm_sampler=LeapHybridCQMSampler()
    sampleset=cqm_sampler.sample_cqm(cqm)

    #We transform it in a panda dataframe
    dataFrame = sampleset.to_pandas_dataframe(sample_column=True)
    dataFrame = dataFrame[['sample','energy','is_feasible']]

    dataFrame = dataFrame.sort_values(by = 'energy')

    #Wa save it
    dataFrame.to_csv("clustering.csv")

    #We return the timer in seconds
    timer = sampleset.info['run_time'] / 1000000
    print("Clustering Done")
    return timer

def VerifClusturing(matrixOfCluster, matrixOfCapacity, matrixOfVolume):
    for i in range(len(matrixOfCluster)):
        capacityTot = 0
        for city in matrixOfCluster[i]:
            capacityTot += matrixOfVolume[city]
        if capacityTot > matrixOfCapacity[i]:
            return False
    return True

def generateClustersFromCSV(numberOfVehicles, numberOfCity):
    """
    Function that read the .csv file of the clusturing to return the list of cluster
    """
    df = pd.read_csv('clustering.csv')
    line = df.loc[df['is_feasible'] == True].iloc[0]
    relation = ast.literal_eval(line['sample'])

    listClusters = []
    for i in range(0,numberOfVehicles):
        #We get every cities of every vehicules, so cluster
        cluster = []
        for j in range(0,numberOfCity):
            keyList = 'x' + str(j) + '_' + str(i)
            if (relation[keyList] == 1):
                cluster.append(j)
        listClusters.append(cluster)

    return listClusters

def generateCostMatrixPerCluster(listClusters, c2):
    """
    Function that generate the cost matrix for every clusters
    """
    costMatrix = []

    for i in range(0,len(listClusters)):
        tmpMatrix = []
        for pos1 in listClusters[i]:
            line = []
            for pos2 in listClusters[i]:
                line.append(c2[pos1][pos2])
            tmpMatrix.append(line)
        costMatrix.append(tmpMatrix)

    return costMatrix

def plotClusters(listCities, listClusters, nameOfpng, timer, showNumber=False):
    """
    Function that save a plot of the clusering of a solution. We can or not show the ID of each cities thanks to the variable showNumber
    """
    plt.figure()

    if (len(listClusters)<11):
        #A set of 10 differents colors for less than 11 clusters
        colors = list(mcolors.TABLEAU_COLORS.values())
    else:
        #We need more colors for each cluster
        colors = list(mcolors.CSS4_COLORS.values())
    
    #For each cluster
    for i in range(0, len(listClusters)):
        #For each city in a cluster
        for j in range(0, len(listClusters[i])):
            #We plot the city with the color defined for the cluster
            plt.scatter(listCities[listClusters[i][j]][0], listCities[listClusters[i][j]][1], c=colors[int(i*len(colors)/len(listClusters))])
            #If showNumber == True, we plot the cities numbers
            if (showNumber):
                plt.annotate(str(listClusters[i][j]), (listCities[listClusters[i][j]][0], listCities[listClusters[i][j]][1]+1))
    
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title("Clustering pour "+str(len(listCities))+" villes\nTemps pour effectuer le clustering : "+str(timer)+"s")
    plt.grid()
    plt.savefig(nameOfpng)
    plt.close()
    return

def TSP (nbOfPoint, matrixOfCost, fileName):
    """
    Return the TSP of the input in the form of a dataframe in a created file with the fileName: 
    "{'x0_0' : 0.0, 'x0_1' : 0.0, ..., x1_0 : 0.0, ..., x{nbOfPoint+1}_{nbOfPoint+1}}", energy, is_feasible
    x{c}_{p} represente the binary of the city c in the position p to know if the city c is in the position p

    This function can be used to find the TSP of an actual problem.
 
    @type  nbOfPoint: int.

    @param nbOfPoint: Number of point that will be sorted to get the shortest distance

    @type  matrixOfCost: List of shape [nbOfPointToCluster, nbOfPointToCluster]
    @param matrixOfCost: Matrix that give the Cost between the point i and the point j,
                           j
                        | ... | i
                        | ... |
                        | ... |
    
    @type  fileName: str

    @param fileName: String that represente the name of the file created at the end of the function

    @rtype:   int

    @return:  the time of the execution of the Quantum TSP.
    """
    
    #Define our model
    cqm=ConstrainedQuadraticModel()

    #Preparation of our matric that will got our solution
    x = {
    (c, p): Binary('x{}_{}'.format(c, p))
    for c in range(nbOfPoint)
    for p in range(nbOfPoint+1)} #+1 cause depository take the first and last position

    #Objective function
    objective = quicksum(matrixOfCost[c1][c2] * x[(c1,p)] * x[(c2,p+1)]
        for c1 in range(nbOfPoint)
        for c2 in range(nbOfPoint)
        for p in range(nbOfPoint) ) #No need to put -1 because we got 1 extra position compare to the number of city

    cqm.set_objective(objective)


    #Constraint
    #The repository need to be at the first and last position
    cqm.add_constraint(x[0,0] == 1)
    cqm.add_constraint(x[0,nbOfPoint] == 1)

    #Every position need to get only 1 city
    for p in range(nbOfPoint):
        cqm.add_constraint(quicksum(x[(c,p)]
        for c in range(nbOfPoint)) == 1)

    #The repository need to have only 2 position
    cqm.add_constraint(quicksum(x[(0,p)]
        for p in range(nbOfPoint+1)) == 2)

    #Every city need to have only 1 position
    for c in range(1,nbOfPoint):
        cqm.add_constraint(quicksum(x[(c,p)]
        for p in range(nbOfPoint)) == 1)

    #Get the solution
    cqm_sampler=LeapHybridCQMSampler()
    sampleset=cqm_sampler.sample_cqm(cqm)

    #Transform the solution in a panda dataframe
    dataFrame = sampleset.to_pandas_dataframe(sample_column=True)
    dataFrame = dataFrame[['sample','energy','is_feasible']]

    dataFrame = dataFrame.sort_values(by = 'energy')

    #Save in a .csv
    dataFrame.to_csv(fileName)

    #Return the timer in seconds
    timer = sampleset.info['run_time'] / 1000000

    return timer

def generateTSPPositionFromCSV(nameOfCSV, clusteurOfCSV):
    """
    Function that read the .csv file of the TSP to return every cities in the good order
    """
    df = pd.read_csv(nameOfCSV)
    line = df.loc[df['is_feasible'] == True].iloc[0]
    relation = ast.literal_eval(line['sample'])

    listPositionsPerCluster = list(np.zeros(len(clusteurOfCSV)).astype(int))

    for i in range(len(clusteurOfCSV)):#For each cities of the cluster
        for j in range(len(clusteurOfCSV)):#For every position in the cluster
            keyList = 'x' + str(i) + '_' + str(j)
            if(relation[keyList] == 1):
                listPositionsPerCluster[j] = int(clusteurOfCSV[i])
    
    return listPositionsPerCluster

def plotTSP(listCities, listPositionsPerCluster, nameOfpng, timer, timerTotal, showNumber=False, showLinkDepot=True):
    """
    Function that save a plot of the TSP of a solution. We can or not show the ID of each cities thanks to the variable 
    showNumber and show the link with the repository thanks to showLinkDepot
    """
    plt.figure()

    colors = list(mcolors.TABLEAU_COLORS.values())
    
    #For each cluster
    for i in range(0, len(listPositionsPerCluster)):
        listPositionsPerCluster[i].append(0)
        #For each city in a cluster
        for j in range(0, len(listPositionsPerCluster[i])):
            #We plot the city with the color defined for the cluster
            plt.scatter(listCities[listPositionsPerCluster[i][j]][0], listCities[listPositionsPerCluster[i][j]][1], c=colors[int(i*len(colors)/len(listPositionsPerCluster))])
            #If showNumber == True, we plot the cities numbers
            if (showNumber):
                plt.annotate(str(listPositionsPerCluster[i][j]), (listCities[listPositionsPerCluster[i][j]][0], listCities[listPositionsPerCluster[i][j]][1]+1))
            #If it is not the last city in the cluster
            if (j < len(listPositionsPerCluster[i])-1):
                #If it is a link with the depot
                if(j == 0 or j == len(listPositionsPerCluster[i])-2):
                    #We don't show the links between the depot and others points
                    if(not showLinkDepot):
                        continue
                #We add an arrow to the graph to link two cities
                plt.annotate("", xy=(listCities[listPositionsPerCluster[i][j]]), xytext=(listCities[listPositionsPerCluster[i][j+1]]), arrowprops=dict(arrowstyle="<-", color=colors[int(i*len(colors)/len(listPositionsPerCluster))], lw=2))

    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title("CVRP pour "+str(len(listCities))+" villes\nTemps pour effectuer le TSP : "+str(timer)+"s\nTemps pour effectuer le CVRP : "+str(timerTotal)+"s")
    plt.grid()
    plt.savefig(nameOfpng)
    plt.close()
    return

def calculateFinalCost(costMatrix, listPositionsPerCluster):
    """
    Function that return the distance total of the input
    """
    cost = 0
    for i in range(0, len(listPositionsPerCluster)):
        for j in range(0, len(listPositionsPerCluster[i])-1):
            cost += costMatrix[listPositionsPerCluster[i][j]][listPositionsPerCluster[i][j+1]]
    return cost

def readVRP(file):
    """
    Function that read the .vrp file of the website http://vrp.atd-lab.inf.puc-rio.br/index.php/en/ 
    and return the list of cities, the list of demand of every cities, the list of capacity of every vehicules and the cost matrix
    """
    f = open(file, "r")

    listCities = []
    listDemand = []
    listVehicles = []

    #We get the number of cities and vehicles from filename
    line = f.readline()
    lineNumbers = re.findall("[\+\-]?[0-9]+", line)
    numberCities = int(lineNumbers[0])
    print(lineNumbers)
    numberVehicles = int(lineNumbers[1])
    #We pass some lines
    for i in range(0,4):
        f.readline()

    #We get the capacity of vehicles
    line = f.readline()
    lineNumbers = re.findall("[\+\-]?[0-9]+", line)
    for i in range(0, numberVehicles):
        listVehicles.append(int(lineNumbers[0]))

    f.readline()

    #For all cities, we store coordinates (x,y)
    for i in range(0, numberCities):
        line = f.readline()
        lineNumbers = re.findall("[\+\-]?[0-9]+", line)
        listCities.append([int(lineNumbers[1]), int(lineNumbers[2])])

    f.readline()

    #For all cities, we store the demand
    for i in range(0, numberCities):
        line = f.readline()
        lineNumbers = re.findall("[\+\-]?[0-9]+", line)
        listDemand.append(int(lineNumbers[1]))

    #Generate Euclidean distances matrix
    costMatrix = generateCostMatrix(listCities)

    return listCities, listDemand, listVehicles, costMatrix

def readSOL(file, numberVehicles):
    """
    Function that read the .sol file of the website http://vrp.atd-lab.inf.puc-rio.br/index.php/en/ and return 
    the supposed optimised solution of our problem
    """
    f = open(file, "r")

    listPositionsPerCluster = []

    for i in range(0, numberVehicles):
        cluster = []
        cluster.append(0)
        line = f.readline()
        lineNumbers = re.findall("[\+\-]?[0-9]+", line)
        for j in range(1, len(lineNumbers)):
            cluster.append(int(lineNumbers[j]))
        cluster.append(0)
        listPositionsPerCluster.append(cluster)
    
    return listPositionsPerCluster

def selfgeneration():
    #Define our problem, the only part you need to change for the problem you want
    numberOfVehicules = 8
    capacityOfCar = [50, 40, 50, 50, 50 ,50 ,50 ,50]
    listTimerCVRP = []
    listTimerCluster = []
    listTimerTSP = []
    listnumberOfCity = []
    numberOfCityMin = 200
    numberOfCityMax = 201
    numberOfCityStep = 1


    for numberOfCity in range (numberOfCityMin,numberOfCityMax,numberOfCityStep):
        #We generate the needed requirement for execute our problem
        listOfCities, c2 = genererateRandomCase (numberOfCity)
        numberOfNodes = numberOfCity + 1 #We have n cities and 1 repository
        volume = [1 for i in range (numberOfCity)]
        volume.insert(0,0) #The repository have no volume
        startCVRP = time.time()

        #We generate our clustering
        ClusterTimer = Classification(numberOfNodes,numberOfVehicules,c2,capacityOfCar,volume)

        #We prepare our cluster for the TSP and to plot them
        listClusters = generateClustersFromCSV(numberOfVehicules, numberOfNodes)
        clusteurCostMatrix = generateCostMatrixPerCluster(listClusters, c2)
        plotClusters(listOfCities,listClusters, "Clusters_"+str(numberOfCity)+".png", np.round(ClusterTimer,2))

        #For each cluster, we do one TSP
        TSPTimer = 0
        for i in range (len(listClusters)):
            TSPTimer += TSP(len(listClusters[i]),clusteurCostMatrix[i], str(i)+".csv")

        listPositionsPerCluster = []
        #We sorted our cities by cluster and by position in this cluster
        for i in range (len(listClusters)):
            listPositionsPerCluster.append(generateTSPPositionFromCSV(str(i)+".csv",listClusters[i]))

        endCVRP = time.time()
        #We plot the final result
        plotTSP(listOfCities,listPositionsPerCluster,"TSP_"+str(numberOfCity)+".png", np.round(TSPTimer,2),np.round(endCVRP-startCVRP,2), False, False)

        listTimerCluster.append(np.round(ClusterTimer,2))
        listTimerCVRP.append(np.round(endCVRP-startCVRP,2))
        listTimerTSP.append(np.round(TSPTimer,2))
        listnumberOfCity.append(numberOfCity)


    #We plot every timer, usefull when we got a lot of data
    plt.figure()
    plt.plot(listnumberOfCity,listTimerCluster)
    plt.ylabel('Temps en s')
    plt.xlabel('Nombre de ville')
    plt.title("Temps d'execution du clustering en fonction du nombre de ville")
    plt.grid()
    plt.savefig("Synchrone temps d'execution du clustering.png")
    plt.close()

    plt.figure()
    plt.plot(listnumberOfCity,listTimerCVRP)
    plt.ylabel('Temps en s')
    plt.xlabel('Nombre de ville')
    plt.title("Temps d'execution du CVRP en fonction du nombre de ville")
    plt.grid()
    plt.savefig("Synchrone temps d'execution du CVRP.png")
    plt.close()

    plt.figure()
    plt.plot(listnumberOfCity,listTimerTSP)
    plt.ylabel('Temps en s')
    plt.xlabel('Nombre de ville')
    plt.title("Temps d'execution du TSP en fonction du nombre de ville")
    plt.grid()
    plt.savefig("Synchrone temps d'execution du TSP.png")
    plt.close()
    return

def literatureGeneration(fileName) :
    startCVRP = time.time()

    #We get the data of the problem
    listCities, listDemand, listVehicles, costMatrix = readVRP(str(fileName)+".vrp")
    numberOfCities = len(listCities)

    #We do the clustering
    ClusterTimer = Classification(numberOfCities, len(listVehicles), costMatrix, listVehicles, listDemand)

    #We prepare our cluster for the TSP and to plot them
    listClusters = generateClustersFromCSV(len(listVehicles), numberOfCities)
    print(VerifClusturing(listClusters,listVehicles,listDemand))
    clusteurCostMatrix = generateCostMatrixPerCluster(listClusters, costMatrix)
    plotClusters(listCities, listClusters, "Clusters_"+fileName+".png", np.round(ClusterTimer,2))



    #For each cluster, we do 1 TSP
    TSPTimer = 0
    for i in range (len(listClusters)):
        TSPTimer += TSP(len(listClusters[i]),clusteurCostMatrix[i], str(i)+".csv")

    listPositionsPerCluster = []
    #We sorted our cities by cluster and by position in this cluster
    for i in range (len(listClusters)):
        listPositionsPerCluster.append(generateTSPPositionFromCSV(str(i)+".csv", listClusters[i]))
    endCVRP = time.time()

    #We plot our final result
    plotTSP(listCities, listPositionsPerCluster, "TSP_"+fileName+".png", np.round(TSPTimer,2), np.round(endCVRP-startCVRP,2), True, True)

    #We calculate and print the final cost of our solution and the one of the optimised solution
    print("Quantum Resolution:", calculateFinalCost(costMatrix, listPositionsPerCluster))
    print("Optimal Resolution:", calculateFinalCost(costMatrix, readSOL(str(fileName)+".sol", len(listVehicles))))
    return


#Main 
literatureGeneration("E-n30-k3")
