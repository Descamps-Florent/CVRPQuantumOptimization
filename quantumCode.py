from dimod import ConstrainedQuadraticModel, CQM, SampleSet
from dimod import Binary, quicksum
from dwave.system import LeapHybridCQMSampler
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
    listCities = [[random.randint(-50,50),random.randint(-50,50)] for i in range(0, nbCities)]

    listCities.insert(0, [0,0])  #On ajoute le dépot en 0,0
    nbCities+=1 #On rajoute le dépot

    #Generate Euclidean distances matrix
    costMatrix = generateCostMatrix(listCities)

    return listCities, costMatrix

def readVRP(file):
    f = open(file, "r")

    listCities = []
    listDemand = []
    listVehicles = []

    #We get the number of cities and vehicles from filename
    line = f.readline()
    lineNumbers = re.findall("[0-9]+", line)
    numberCities = int(lineNumbers[0])
    numberVehicles = int(lineNumbers[1])
    
    #We pass some lines
    for i in range(0,4):
        f.readline()

    #We get the capacity of vehicles
    line = f.readline()
    lineNumbers = re.findall("[0-9]+", line)
    for i in range(0, numberVehicles):
        listVehicles.append(int(lineNumbers[0]))

    f.readline()

    #For all cities, we store coordinates (x,y)
    for i in range(0, numberCities):
        line = f.readline()
        line = line.split(" ")
        listCities.append([int(line[1]), int(line[2])])

    f.readline()

    #For all cities, we store the demand
    for i in range(0, numberCities):
        line = f.readline()
        line = line.split(" ")
        listDemand.append(int(line[1]))

    #Generate Euclidean distances matrix
    costMatrix = generateCostMatrix(listCities)

    return listCities, listDemand, listVehicles, costMatrix

def generateCostMatrix(listCities):
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
 
    This function can be used to fin the clustering of an actual problem.
 
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

    @rtype:   panda dataframe

    @return:  the clusturing of the project.
    """
    
    #Définition de notre model
    cqm=ConstrainedQuadraticModel()

    #Préparation de la forme de notre matrice qui accueillera la solution
    x = {
    (i, d): Binary('x{}_{}'.format(i, d))
    for i in range(nbOfPointToCluster)
    for d in range(nbOfCluster)}

    #Fonction objectif
    objective = quicksum(matrixOfCost[i][j] * x[(i,d)] * x[(j,d)]
        for i in range(nbOfPointToCluster)
        for j in range(i+1, nbOfPointToCluster)
        for d in range(nbOfCluster) )

    cqm.set_objective(objective)


    #Contraintes
    #On veut que le dépot soit dans tous les clusteurs
    for d in range(nbOfCluster):
        cqm.add_constraint(x[(0,d)] == 1)

    #Tous les points doivent être dans un seul et unique cluster, sauf le dépot qui est dans tous les clusteurs
    for i in range(1,nbOfPointToCluster):
        cqm.add_constraint(quicksum(x[(i,d)]
        for d in range(nbOfCluster)) == 1)

    #Le nombre de point combiner à leur cout ne doit pas être supèrieur à la capacité d'un cluster.
    for d in range(nbOfCluster):
        cqm.add_constraint(quicksum(matrixOfVolume[i] * x[(i,d)]
        for i in range(nbOfPointToCluster)) <= matrixOfCapacity[d])

    cqm_sampler=LeapHybridCQMSampler()
    sampleset=cqm_sampler.sample_cqm(cqm)

    dataFrame = sampleset.to_pandas_dataframe(sample_column=True)
    dataFrame = dataFrame[['sample','energy','is_feasible']]

    dataFrame = dataFrame.sort_values(by = 'energy')

    dataFrame.to_csv("clustering.csv")

    timer = sampleset.info['run_time'] / 1000000 #On transforme en secondes
    return timer

def generateClustersFromCSV(numberOfVehicles, numberOfCity):
    df = pd.read_csv('clustering.csv')
    line = df.loc[df['is_feasible'] == True].iloc[0]
    relation = ast.literal_eval(line['sample'])

    listClusters = []

    for i in range(0,numberOfVehicles):
        cluster = []
        for j in range(0,numberOfCity):
            keyList = 'x' + str(j) + '_' + str(i)
            if (relation[keyList] == 1):
                cluster.append(j)
        listClusters.append(cluster)

    return listClusters

def generateCostMatrixByCluster(listClusters, c2):
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

    
    #Définition de notre model
    cqm=ConstrainedQuadraticModel()

    #Préparation de la forme de notre matrice qui accueillera la solution
    x = {
    (c, p): Binary('x{}_{}'.format(c, p))
    for c in range(nbOfPoint)
    for p in range(nbOfPoint+1)} #+1 car le dépot prend la position 0 et la dernière

    #Fonction objectif
    objective = quicksum(matrixOfCost[c1][c2] * x[(c1,p)] * x[(c2,p+1)]
        for c1 in range(nbOfPoint)
        for c2 in range(nbOfPoint)
        for p in range(nbOfPoint) ) #Pas besoin de mettre -1 car on a 1 position en plus que le nombre de villes

    cqm.set_objective(objective)


    #Contraintes
    # Le dépot doit être a la première et dernière position possible
    cqm.add_constraint(x[0,0] == 1)
    cqm.add_constraint(x[0,nbOfPoint] == 1)

    #Chaque position doit contenir seulement une ville
    for p in range(nbOfPoint):
        cqm.add_constraint(quicksum(x[(c,p)]
        for c in range(nbOfPoint)) == 1)

    #Le dépot doit avoir 2 positions
    cqm.add_constraint(quicksum(x[(0,p)]
        for p in range(nbOfPoint+1)) == 2)

    #Chaque ville doit avoir une seule position
    for c in range(1,nbOfPoint):
        cqm.add_constraint(quicksum(x[(c,p)]
        for p in range(nbOfPoint)) == 1)

    cqm_sampler=LeapHybridCQMSampler()
    sampleset=cqm_sampler.sample_cqm(cqm)

    dataFrame = sampleset.to_pandas_dataframe(sample_column=True)
    dataFrame = dataFrame[['sample','energy','is_feasible']]

    dataFrame = dataFrame.sort_values(by = 'energy')

    dataFrame.to_csv(fileName)

    timer = sampleset.info['run_time'] / 1000000 #On transforme en secondes

    return timer

def generateTSPPositionFromCSV(nameOfCSV, clusteurOfCSV):
    df = pd.read_csv(nameOfCSV)
    line = df.loc[df['is_feasible'] == True].iloc[0]
    relation = ast.literal_eval(line['sample'])

    listPositionsPerCluster = list(np.zeros(len(clusteurOfCSV)).astype(int))

    for i in range(len(clusteurOfCSV)):#Pour toute les villes du clusteurs
        for j in range(len(clusteurOfCSV)):#Pour toute les positions du clusteurs
            keyList = 'x' + str(i) + '_' + str(j)
            if(relation[keyList] == 1):
                listPositionsPerCluster[j] = int(clusteurOfCSV[i])
    
    return listPositionsPerCluster

def plotTSP(listCities, listPositionsPerCluster, nameOfpng, timer, timerTotal, showNumber=False, showLinkDepot=True):
    plt.figure()

    if (len(listClusters)<11):
        #A set of 10 differents colors for less than 11 clusters
        colors = list(mcolors.TABLEAU_COLORS.values())
    else:
        #We need more colors for each cluster
        colors = list(mcolors.CSS4_COLORS.values())
    
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


"""
#Définition des variables du problème
numberOfVehicules = 8
capacityOfCar = [50, 40, 50, 50, 50 ,50 ,50 ,50]
listTimerCVRP = []
listTimerCluster = []
listTimerTSP = []
listnumberOfCity = []

for numberOfCity in range (200,201):
    listOfCities, c2 = genererateRandomCase (numberOfCity)
    numberOfNodes = numberOfCity + 1 #On a n villes et 1 dépot
    volume = [1 for i in range (numberOfCity)]
    volume.insert(0,0) #Le dépot n'a pas de poids
    startCVRP = time.time()


    ClusterTimer = Classification(numberOfNodes,numberOfVehicules,c2,capacityOfCar,volume)


    listClusters = generateClustersFromCSV(numberOfVehicules, numberOfNodes)
    clusteurCostMatrix = generateCostMatrixByCluster(listClusters, c2)

    plotClusters(listOfCities,listClusters, "Clusters_"+str(numberOfCity)+".png", np.round(ClusterTimer,2))

    TSPTimer = 0
    for i in range (len(listClusters)):
        TSPTimer += TSP(len(listClusters[i]),clusteurCostMatrix[i], str(i)+".csv")

    listPositionsPerCluster = []

    for i in range (len(listClusters)):
        listPositionsPerCluster.append(generateTSPPositionFromCSV(str(i)+".csv",listClusters[i]))

    endCVRP = time.time()
    plotTSP(listOfCities,listPositionsPerCluster,"TSP_"+str(numberOfCity)+".png", np.round(TSPTimer,2),np.round(endCVRP-startCVRP,2))

    listTimerCluster.append(np.round(ClusterTimer,2))
    listTimerCVRP.append(np.round(endCVRP-startCVRP,2))
    listTimerTSP.append(np.round(TSPTimer,2))
    listnumberOfCity.append(numberOfCity)


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
"""

startCVRP = time.time()

listCities, listDemand, listVehicles, costMatrix = readVRP("../E/E-n22-k4.vrp")
numberOfCities = len(listCities)

ClusterTimer = Classification(numberOfCities, len(listVehicles), costMatrix, listVehicles, listDemand)

listClusters = generateClustersFromCSV(len(listVehicles), numberOfCities)
clusteurCostMatrix = generateCostMatrixByCluster(listClusters, costMatrix)

plotClusters(listCities, listClusters, "Clusters_"+str(numberOfCities)+".png", np.round(ClusterTimer,2))

TSPTimer = 0
for i in range (len(listClusters)):
    TSPTimer += TSP(len(listClusters[i]),clusteurCostMatrix[i], str(i)+".csv")

listPositionsPerCluster = []

for i in range (len(listClusters)):
    listPositionsPerCluster.append(generateTSPPositionFromCSV(str(i)+".csv", listClusters[i]))
endCVRP = time.time()

plotTSP(listCities, listPositionsPerCluster, "TSP_"+str(numberOfCities)+".png", np.round(TSPTimer,2), np.round(endCVRP-startCVRP,2))
