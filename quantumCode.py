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

def genererateRandomCase(nbCities):
    listCities = [[random.randint(-10000,10000),random.randint(-10000,10000)] for i in range(0, nbCities)]
    costMatrix = []
    
    for i in range(0, nbCities):
        lineCostMatrix = []
        for j in range(0, nbCities):
            lineCostMatrix.append(distance.euclidean(listCities[i], listCities[j]))
        costMatrix.append(lineCostMatrix)

    return listCities, costMatrix

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
    #Le nombre de point combiner à leur cout ne doit pas être supèrieur à la capacité d'un cluster.
    for d in range(nbOfCluster):
        cqm.add_constraint(quicksum(matrixOfVolume[i] * x[(i,d)]
        for i in range(nbOfPointToCluster)) <= matrixOfCapacity[d])

    #Tous les points doivent être dans un cluster
    for i in range(nbOfPointToCluster):
        cqm.add_constraint(quicksum(x[(i,d)]
        for d in range(nbOfCluster)) == 1)

    cqm_sampler=LeapHybridCQMSampler()
    sampleset=cqm_sampler.sample_cqm(cqm)

    run_time = sampleset.info['run_time']
    qpu_time = sampleset.info['qpu_access_time']
    print("Run Time for Classification", run_time, "µs")
    print("QPU Time for Classification", qpu_time, "µs")
    print(cqm_sampler.properties)

    dataFrame = sampleset.to_pandas_dataframe(sample_column=True)
    dataFrame = dataFrame[['sample','energy','is_feasible']]

    dataFrame = dataFrame.sort_values(by = 'energy')

    dataFrame.to_csv("clustering.csv")

    return dataFrame, run_time, qpu_time

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

def plotClusters(listCities, listClusters, nameOfpng):
    plt.figure()
    #mcolors.CSS4_COLORS if more than 10 clusters to plot
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i in range(0, len(listClusters)):
        for j in range(0, len(listClusters[i])):
            plt.scatter(listCities[listClusters[i][j]][0], listCities[listClusters[i][j]][1], c=colors[int(i*len(colors)/len(listClusters))])
            plt.annotate(str(listClusters[i][j]), (listCities[listClusters[i][j]][0], listCities[listClusters[i][j]][1]+1))
    
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.grid()
    plt.savefig(nameOfpng)

    return

def TSP (nbOfPoint, matrixOfCost, fileName):

    
    #Définition de notre model
    cqm=ConstrainedQuadraticModel()

    #Préparation de la forme de notre matrice qui accueillera la solution
    x = {
    (c, p): Binary('x{}_{}'.format(c, p))
    for c in range(nbOfPoint)
    for p in range(nbOfPoint)}

    #Fonction objectif
    objective = quicksum(matrixOfCost[c1][c2] * x[(c1,p)] * x[(c2,p+1)]
        for c1 in range(nbOfPoint)
        for c2 in range(nbOfPoint)
        for p in range(nbOfPoint-1) )

    cqm.set_objective(objective)


    #Contraintes
    #Chaque position doit avoir une seule ville
    for p in range(nbOfPoint):
        cqm.add_constraint(quicksum(x[(c,p)]
        for c in range(nbOfPoint)) == 1)

    #Chaque ville doit avoir une seule position
    for c in range(nbOfPoint):
        cqm.add_constraint(quicksum(x[(c,p)]
        for p in range(nbOfPoint)) == 1)

    cqm_sampler=LeapHybridCQMSampler()
    sampleset=cqm_sampler.sample_cqm(cqm)
    print("Run Time for TSP", sampleset.info['run_time'], "µs")
    print("QPU Time for TSP", sampleset.info['qpu_access_time'], "µs")

    dataFrame = sampleset.to_pandas_dataframe(sample_column=True)
    dataFrame = dataFrame[['sample','energy','is_feasible']]

    dataFrame = dataFrame.sort_values(by = 'energy')

    dataFrame.to_csv(fileName)

    return dataFrame

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

def plotTSP(listCities, listPositionsPerCluster, nameOfpng):
    plt.figure()
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i in range(0, len(listPositionsPerCluster)):
        for j in range(0, len(listPositionsPerCluster[i])):
            plt.scatter(listCities[listPositionsPerCluster[i][j]][0], listCities[listPositionsPerCluster[i][j]][1], c=colors[int(i*len(colors)/len(listPositionsPerCluster))])
            plt.annotate(str(listPositionsPerCluster[i][j]), (listCities[listPositionsPerCluster[i][j]][0], listCities[listPositionsPerCluster[i][j]][1]+1))
            if (j < len(listPositionsPerCluster[i])-1):
                plt.annotate("", xy=(listCities[listPositionsPerCluster[i][j]]), xytext=(listCities[listPositionsPerCluster[i][j+1]]), arrowprops=dict(arrowstyle="<-", color=colors[int(i*len(colors)/len(listPositionsPerCluster))], lw=2))

    plt.ylabel('Y')
    plt.xlabel('X')
    plt.grid()
    plt.savefig(nameOfpng)

    return

#Définition des variables du problème
classification_run_time = []
classification_qpu_time = []

for numberOfCity in range (10,101, 10):
    numberOfVehicules = 3
    listOfCities, c2 = genererateRandomCase (numberOfCity)
    capacityOfCar = [30, 100, 50]
    volume = [1 for i in range (numberOfCity)]

    clusteur, run_time, qpu_time = Classification(numberOfCity,numberOfVehicules,c2,capacityOfCar,volume)
    classification_run_time.append(run_time)
    classification_qpu_time.append(qpu_time)

    listClusters = generateClustersFromCSV(numberOfVehicules, numberOfCity)
    print(listClusters)
    clusteurCostMatrix = generateCostMatrixByCluster(listClusters, c2)
    print(clusteurCostMatrix)
    plotClusters(listOfCities,listClusters, "Clusters_"+str(numberOfCity)+".png")

    for i in range (len(listClusters)):
        TSP(len(listClusters[i]),clusteurCostMatrix[i], str(i)+".csv")

    listPositionsPerCluster = []

    for i in range (len(listClusters)):
        listPositionsPerCluster.append(generateTSPPositionFromCSV(str(i)+".csv",listClusters[i]))

    print(listPositionsPerCluster)
    plotTSP(listOfCities,listPositionsPerCluster,"TSP_"+str(numberOfCity)+".png")

print(classification_run_time)
plt.figure()
plt.plot([i for i in range(10,101,10)], classification_run_time)
plt.savefig("run_time.png")
plt.close()

plt.figure()
plt.plot([i for i in range(10,101,10)], classification_qpu_time)
plt.savefig("qpu_time.png")
plt.close()
