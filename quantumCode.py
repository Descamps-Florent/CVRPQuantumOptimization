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
import time
import re
import os

# To display the mapping on the qubit and other data
# import dwave.inspector as inspector



def genererateRandomCase(nbCities):
    """
    Function that generates a random case of CVRP and return the list of the cities and it's cost matrix
    """
    listCities = [[random.randint(-50,50),random.randint(-50,50)] for i in range(0, nbCities)]

    listCities.insert(0, [0,0])  #We add the depot in (0,0)
    nbCities+=1 #We add the depot

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
            lineCostMatrix.append(round(math.sqrt((listCities[i][0]-listCities[j][0])**2 + (listCities[i][1]-listCities[j][1])**2)))
        costMatrix.append(lineCostMatrix)
                    
    return costMatrix


def plotCostMatrix(costMatrix):
    cityNumber = [i for i in range(0,len(costMatrix))]
    df = pd.DataFrame(costMatrix, columns=cityNumber, index=cityNumber)
    try:
        text_file = open("CostMatrix.txt", "w")
    except OSError:
        print("Could not open/read file:", file)
        exit()
    dfAsString = df.to_string(header=True, index=True)
    text_file.write(dfAsString)
    text_file.close()
    
    return









# --------------------------------------------------------------------------------------------- #
#                                     CLUSTERING                                                #
# --------------------------------------------------------------------------------------------- #
def Classification (nbOfPointToCluster, nbOfCluster, matrixOfCost, vectorOfCapacity, vectorOfVolume):
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
    @type  vectorOfCapacity: List of shape [nbOfCluster]
    @param vectorOfCapacity: Matrix that give the Capacity of each cluster
    @type  vectorOfVolume: List of shape [nbOfPointToCluster]
    @param vectorOfVolume: Matrix that give the cost of each point of the city
    @rtype:   int
    @return:  the timer of the clustering.
    """
    #Define our model
    cqm=ConstrainedQuadraticModel()


   #Preparation of our matrix that will got our solution
    x = {
    (i, d): Binary('x{}_{}'.format(i, d))
    for i in range(nbOfPointToCluster)
    for d in range(nbOfCluster)}


    # ------------------------------------------------------------------------------------ #
    #                                 Objective function:                                  #
    # ------------------------------------------------------------------------------------ #
    objective = quicksum(matrixOfCost[i][j] * x[(i,d)] * x[(j,d)]
        for i in range(nbOfPointToCluster)
        for j in range(i+1, nbOfPointToCluster)
        for d in range(nbOfCluster) )

    cqm.set_objective(objective)



    # ------------------------------------------------------------------------------------ #
    #                               subject to the constraints:                            #
    # ------------------------------------------------------------------------------------ #
    #We want the depot in every cluster
    for d in range(nbOfCluster):
        cqm.add_constraint(x[(0,d)] == 1)

    #The sum of the capacity require by the point should not exceed the total capacity of the cluster
   
   
    print('nbOfPointToCluster : ', nbOfPointToCluster)
    print('nbOfCluster : ', nbOfCluster)
    for d in range(nbOfCluster):
        cqm.add_constraint(quicksum(vectorOfVolume[i] * x[(i,d)]
        for i in range(nbOfPointToCluster)) <= vectorOfCapacity[d])

    #Every point should be in 1 and only 1 cluster except the depot
    for i in range(1,nbOfPointToCluster):
        cqm.add_constraint(quicksum(x[(i,d)]
        for d in range(nbOfCluster)) == 1)



    # ------------------------------------------------------------------------------------ #
    #                               Resolution & data analysis:                            #
    # ------------------------------------------------------------------------------------ #
    #We get our solution
    cqm_sampler=LeapHybridCQMSampler()
    sampleset=cqm_sampler.sample_cqm(cqm)

    #We transform it in a panda dataframe
    dataFrame = sampleset.to_pandas_dataframe(sample_column=True)
    dataFrame = dataFrame[['sample','energy','is_feasible']]
    dataFrame = dataFrame.sort_values(by = 'energy')
    dataFrame.to_csv("clustering.csv")

    #We return the timer in seconds
    timer = sampleset.info['run_time'] / 1000000
    print("Clustering Done")
    return timer

 




# --------------------------------------------------------------------------------------------- #
#                                     VerifClusturing                                           #
# --------------------------------------------------------------------------------------------- #
def VerifClusturing(matrixOfCluster, vectorOfCapacity, vectorOfVolume):
    for i in range(len(matrixOfCluster)):
        capacityTot = 0
        for city in matrixOfCluster[i]:
            capacityTot += vectorOfVolume[city]
        if capacityTot > vectorOfCapacity[i]:
            return False
    return True








# --------------------------------------------------------------------------------------------- #
#                                         generateClustersFromCSV                               #
# --------------------------------------------------------------------------------------------- #
def generateClustersFromCSV(numberOfVehicles, numberOfCities):
    """
    Function that read the .csv file of the clusturing to return the list of cluster
    """
    df = pd.read_csv('clustering.csv')
    line = df.loc[df['is_feasible'] == True].iloc[0]
    relation = ast.literal_eval(line['sample'])

    listClusters = []
    for i in range(0,numberOfVehicles):
        #We get every cities of every vehicles, so cluster
        cluster = []
        for j in range(0,numberOfCities):
            keyList = 'x' + str(j) + '_' + str(i)
            if (relation[keyList] == 1):
                cluster.append(j)
        listClusters.append(cluster)

    return listClusters



# --------------------------------------------------------------------------------------------- #
#                                     generateCostMatrixPerCluster                              #
# --------------------------------------------------------------------------------------------- #
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





# --------------------------------------------------------------------------------------------- #
#                                     plotClusters                                              #
# --------------------------------------------------------------------------------------------- #
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









# --------------------------------------------------------------------------------------------- #
#                                            TSP                                                #
# --------------------------------------------------------------------------------------------- #
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



    #Preparation of our variables
    x = {
    (c, p): Binary('x{}_{}'.format(c, p))
    for c in range(nbOfPoint)
    for p in range(nbOfPoint+1)} #+1 cause depository take the first and last position



    #Objective function
    objective = quicksum(matrixOfCost[c1][c2] * x[(c1,p)] * x[(c2,p+1)]
        for c1 in range(nbOfPoint)
        for c2 in range(nbOfPoint)
        for p in range(nbOfPoint) ) 
        #No need to put -1 because we got 1 extra position compare to the number of city
    cqm.set_objective(objective)




    #Constraints
    #The depot need to be at the first and last position
    cqm.add_constraint(x[0,0] == 1)
    cqm.add_constraint(x[0,nbOfPoint] == 1)
    #The depot need to have only 2 positions (to update...)
    cqm.add_constraint(quicksum(x[(0,p)]
        for p in range(nbOfPoint+1)) == 2)
 
    #Every position needs to get only 1 city
    for p in range(nbOfPoint):
        cqm.add_constraint(quicksum(x[(c,p)]
        for c in range(nbOfPoint)) == 1)

    #Every city needs to have only 1 position
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









# --------------------------------------------------------------------------------------------- #
#                                     VerifClusturing                                           #
# --------------------------------------------------------------------------------------------- #
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
    
    listPositionsPerCluster.append(0)
    
    return listPositionsPerCluster







# --------------------------------------------------------------------------------------------- #
#                                     plotTSP                                                   #
# --------------------------------------------------------------------------------------------- #
def plotTSP(listCities, listPositionsPerCluster, nameOfpng, timer, timerTotal, showNumber=False, showLinkDepot=True):
    """
    Function that save a plot of the TSP of a solution. We can or not show the ID of each cities thanks to the variable 
    showNumber and show the link with the depot thanks to showLinkDepot
    """
    plt.figure()

    colors = list(mcolors.TABLEAU_COLORS.values())
    
    #For each cluster
    for i in range(0, len(listPositionsPerCluster)):
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








# --------------------------------------------------------------------------------------------- #
#                                     calculateFinalCost                                        #
# --------------------------------------------------------------------------------------------- #
def calculateFinalCost(costMatrix, listPositionsPerCluster):
    """
    Function that return the distance total of the input
    """
    cost = 0
    for i in range(0, len(listPositionsPerCluster)):
        for j in range(0, len(listPositionsPerCluster[i])-1):
            cost += costMatrix[listPositionsPerCluster[i][j]][listPositionsPerCluster[i][j+1]]
    return cost













# --------------------------------------------------------------------------------------------- #
#                                     readVRP                                                   #
# --------------------------------------------------------------------------------------------- #
def readVRP(file):
    """
    Function that read the .vrp file of the website http://vrp.atd-lab.inf.puc-rio.br/index.php/en/ 
    and return the list of cities, the list of demand of every cities, the list of capacity of every vehicles and the cost matrix
    """
    try:
        f = open(file, "r")
    except OSError:
        print("Could not open/read file:", file)
        exit()

    listCities = []
    listDemand = []
    listVehicles = []

    #We get the number of cities and vehicles from filename
    line = f.readline()
    lineNumbers = re.findall("[\+\-]?[0-9]+", line)
    numberCities = int(lineNumbers[0])
    
    numberVehicles = int(lineNumbers[1])
    #We pass some lines until Capacity section
    while ("CAPACITY : " in line) == 0:
        line = f.readline()

    #We get the capacity of vehicles
    lineNumbers = re.findall("[\+\-]?[0-9]+", line)
    for i in range(0, numberVehicles):
        listVehicles.append(int(lineNumbers[0]))

    while ("NODE_COORD_SECTION" in line) == 0:
        line = f.readline()

    #For all cities, we store coordinates (x,y)
    for i in range(0, numberCities):
        line = f.readline()
        lineNumbers = re.findall("[\+\-]?[0-9]+[.]?[0-9]*", line)
        listCities.append([float(lineNumbers[1]), float(lineNumbers[2])])

    f.readline()

    #For all cities, we store the demand
    for i in range(0, numberCities):
        line = f.readline()
        lineNumbers = re.findall("[\+\-]?[0-9]+", line)
        listDemand.append(int(lineNumbers[1]))

    #Generate Euclidean distances matrix
    costMatrix = generateCostMatrix(listCities)

    return listCities, listDemand, listVehicles, costMatrix











# --------------------------------------------------------------------------------------------- #
#                                     readVRPWithoutListCities                                  #
# --------------------------------------------------------------------------------------------- #
def readVRPWithoutListCities(file):
    """
    Function that read the .vrp file of the website http://vrp.atd-lab.inf.puc-rio.br/index.php/en/ 
    and return the list of cities, the list of demand of every cities, the list of capacity of every vehicles and the cost matrix
    """
    try:
        f = open(file, "r")
    except OSError:
        print("Could not open/read file:", file)
        exit()

    costMatrix = []
    listDemand = []
    listVehicles = []

    #We get the number of cities and vehicles from filename
    line = f.readline()
    lineNumbers = re.findall("[\+\-]?[0-9]+", line)
    numberCities = int(lineNumbers[0])
    
    numberVehicles = int(lineNumbers[1])
    #We pass some lines until Capacity section
    while ("CAPACITY : " in line) == 0:
        line = f.readline()

    #We get the capacity of vehicles
    lineNumbers = re.findall("[\+\-]?[0-9]+", line)
    for i in range(0, numberVehicles):
        listVehicles.append(int(lineNumbers[0]))

    while ("EDGE_WEIGHT_SECTION" in line) == 0:
        line = f.readline()
    listUnorderedCost = []

    while ("DEMAND_SECTION" in line) == 0:
        line = f.readline()
        costsInsideLine = re.findall("[\+\-]?[0-9]+[.]?[0-9]*", line)
        for i in range(0,len(costsInsideLine)):
            listUnorderedCost.append(costsInsideLine[i])
    
    lineMatrix = []

    #We store each weigth in a cost matrix
    for i in range(0, numberCities):
        lineMatrix.clear()
        for j in range(0, numberCities):
            if i>j:
                lineMatrix.append(int(listUnorderedCost.pop(0)))
            else:
                lineMatrix.append(0)
        #We add a new line in the matrix
        costMatrix.append(lineMatrix[:])
    
    #We copy the values ​​from the lower triangular matrix to the upper triangular matrix
    for i in range(1, numberCities):
        for j in range(0,i):
            costMatrix[j][i] = costMatrix[i][j]



    #For all cities, we store the demand
    for i in range(0, numberCities):
        line = f.readline()
        lineNumbers = re.findall("[\+\-]?[0-9]+", line)
        listDemand.append(int(lineNumbers[1]))


    return listDemand, listVehicles, costMatrix














# --------------------------------------------------------------------------------------------- #
#                                     readSOL                                                   #
# --------------------------------------------------------------------------------------------- #
def readSOL(file):
    """
    Function that read the .sol file of the website http://vrp.atd-lab.inf.puc-rio.br/index.php/en/ and return 
    the supposed optimised solution of our problem
    """
    try:
        f = open(file, "r")
    except OSError:
        print("Could not open/read file:", file)
        exit()

    listPositionsPerCluster = []

    line = f.readline()

    while "Route" in line:
        cluster = []
        cluster.append(0)
        lineNumbers = re.findall("[\+\-]?[0-9]+", line)
        for j in range(1, len(lineNumbers)):
            cluster.append(int(lineNumbers[j]))
        cluster.append(0)
        listPositionsPerCluster.append(cluster)
        line = f.readline()
    
    return listPositionsPerCluster











# --------------------------------------------------------------------------------------------- #
#                                     selfgeneration                                            #
# --------------------------------------------------------------------------------------------- #
def selfgeneration(numberOfVehicles, numberOfCities, capaConsumptionMin, capaConsumptionMax):
    #Define our problem, the only part you need to change for the problem you want
    
    # To Erase    capacityOfVehicles = [50, 40, 50, 50, 50 ,50 ,50 ,50]
    
    
    listTimerCVRP = []
    listTimerCluster = []
    listTimerTSP = []
    listnumberOfCities = []

    #We generate randomly the capacity of vehicles
    capacityOfVehiclesInt = int(math.ceil( (capaConsumptionMax * numberOfCities) / numberOfVehicles))
    capacityOfVehicles = [capacityOfVehiclesInt for i in range(numberOfVehicles)]



    #To ERASE
    """
    numberOfCitiesMin = 200
    numberOfCitiesMax = 201
    numberOfCitiesStep = 1
    """
    #To ERASE for numberOfCities in range (numberOfCitiesMin,numberOfCitiesMax,numberOfCitiesStep):


    #We generate the needed requirement for execute our problem
    #The cities and the costMatrix c2
    listOfCities, c2 = genererateRandomCase (numberOfCities)
    numberOfNodes = numberOfCities + 1 #We have n cities and 1 depot

    # We add the capacity consumption of each package/city

    
 
    volume = []
    for i in range(numberOfCities):
        n = random.randint(capaConsumptionMin,capaConsumptionMax)
        volume.append(n)

    print("random capacityOfVehicles : ", capacityOfVehicles)
    print("numberOfCities:", numberOfCities)
    print(capacityOfVehicles)
    print("random capa consumption ")
    print(volume)
    print("numberOfNodes :", numberOfNodes)

    # TO ERASE    volume = [1 for i in range (numberOfCities)]
    
    volume.insert(0,0) #The depot have no volume



 

    startCVRP = time.time()

    #We generate our clustering
    ClusterTimer = Classification(numberOfCities,len(capacityOfVehicles),c2,capacityOfVehicles,volume)
   
    #We prepare our cluster for the TSP and to plot them
    listClusters = generateClustersFromCSV(len(capacityOfVehicles), numberOfCities)

    
    print('Feasible? => ', VerifClusturing(listClusters,capacityOfVehicles,volume))


    clusteurCostMatrix = generateCostMatrixPerCluster(listClusters, c2)
    plotClusters(listOfCities,listClusters, "Clusters_"+str(numberOfCities)+".png", np.round(ClusterTimer,2))

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
    plotTSP(listOfCities,listPositionsPerCluster,"TSP_"+str(numberOfCities)+".png", np.round(TSPTimer,2),np.round(endCVRP-startCVRP,2), True, True)

    listTimerCluster.append(np.round(ClusterTimer,2))
    listTimerCVRP.append(np.round(endCVRP-startCVRP,2))
    listTimerTSP.append(np.round(TSPTimer,2))
    listnumberOfCities.append(numberOfCities)
 
    #We plot every timer, usefull when we got a lot of data
    plt.figure()
    plt.plot(listnumberOfCities,listTimerCluster)
    plt.ylabel('Temps en s')
    plt.xlabel('Nombre de ville')
    plt.title("Temps d'execution du clustering en fonction du nombre de ville")
    plt.grid()
    plt.savefig("Synchrone temps d'execution du clustering.png")
    plt.close()

    plt.figure()
    plt.plot(listnumberOfCities,listTimerCVRP)
    plt.ylabel('Temps en s')
    plt.xlabel('Nombre de ville')
    plt.title("Temps d'execution du CVRP en fonction du nombre de ville")
    plt.grid()
    plt.savefig("Synchrone temps d'execution du CVRP.png")
    plt.close()

    plt.figure()
    plt.plot(listnumberOfCities,listTimerTSP)
    plt.ylabel('Temps en s')
    plt.xlabel('Nombre de ville')
    plt.title("Temps d'execution du TSP en fonction du nombre de ville")
    plt.grid()
    plt.savefig("Synchrone temps d'execution du TSP.png")
    plt.close()

    # We calculate and print the final cost of our solution and the one of the optimised solution
    print("D-Wave Hybrid Resolution:", calculateFinalCost(c2, listPositionsPerCluster))
 

 
    return








 





# --------------------------------------------------------------------------------------------- #
#                                     literatureGeneration                                      #
# --------------------------------------------------------------------------------------------- #
def literatureGeneration(path_to_file, fileName, path_to_png) :
    print("CVRP for ",fileName)
    startCVRP = time.time()

    #We get the data of the problem
    listCities, listDemand, listVehicles, costMatrix = readVRP(str(path_to_file+fileName)+".vrp")
    plotCostMatrix(costMatrix)
    numberOfCities = len(listCities)


    #                     ------- clustering -------
    #We do the clustering
    ClusterTimer = Classification(numberOfCities, len(listVehicles), costMatrix, listVehicles, listDemand)

    #We prepare our cluster for the TSP and to plot them
    listClusters = generateClustersFromCSV(len(listVehicles), numberOfCities)
 
    clusteurCostMatrix = generateCostMatrixPerCluster(listClusters, costMatrix)
    plotClusters(listCities, listClusters, path_to_png+"Clusters_"+fileName+".png", np.round(ClusterTimer,2))


    #                         ------- TSP -------
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
    plotTSP(listCities, listPositionsPerCluster, path_to_png+"TSP_"+fileName+".png", np.round(TSPTimer,2), np.round(endCVRP-startCVRP,2), True, True)

    #We plot the solution result
    plotTSP(listCities, readSOL(str(path_to_file+fileName)+".sol"), path_to_png+"TSP_SOL_"+fileName+".png", np.round(TSPTimer,2), np.round(endCVRP-startCVRP,2), True, True)


    Quantum_Resolution = calculateFinalCost(costMatrix, listPositionsPerCluster)
    OptimalResolution = calculateFinalCost(costMatrix, readSOL(str(path_to_file+fileName)+".sol"))

    #We calculate and print the final cost of our solution and the one of the optimised solution
    print("Total time :", endCVRP-startCVRP, "Clusturing time : ", ClusterTimer, "TSP time : ",TSPTimer)
    print("Quantum Resolution:", Quantum_Resolution)
    print("Optimal Resolution:", OptimalResolution)

    return (fileName, endCVRP-startCVRP, ClusterTimer, TSPTimer, Quantum_Resolution, OptimalResolution)
 






# --------------------------------------------------------------------------------------------- #
#                               literatureGenerationWithoutListCities                           #
# --------------------------------------------------------------------------------------------- #
def literatureGenerationWithoutListCities(path_to_file, fileName) :
    print("CVRP for ",fileName)
    startCVRP = time.time()

    #We get the data of the problem
    listDemand, listVehicles, costMatrix = readVRPWithoutListCities(str(path_to_file+fileName)+".vrp")
    plotCostMatrix(costMatrix)
    numberOfCities = len(costMatrix[0])


    #                     ------- clustering -------
    #We do the clustering
    ClusterTimer = Classification(numberOfCities, len(listVehicles), costMatrix, listVehicles, listDemand)

    #We prepare our cluster for the TSP and to plot them
    listClusters = generateClustersFromCSV(len(listVehicles), numberOfCities)
 
    clusteurCostMatrix = generateCostMatrixPerCluster(listClusters, costMatrix)


    #                         ------- TSP -------
    #For each cluster, we do 1 TSP
    TSPTimer = 0
    for i in range (len(listClusters)):
        TSPTimer += TSP(len(listClusters[i]),clusteurCostMatrix[i], str(i)+".csv")



    listPositionsPerCluster = []
    #We sorted our cities by cluster and by position in this cluster
    for i in range (len(listClusters)):
        listPositionsPerCluster.append(generateTSPPositionFromCSV(str(i)+".csv", listClusters[i]))
    endCVRP = time.time()


    Quantum_Resolution = calculateFinalCost(costMatrix, listPositionsPerCluster)
    OptimalResolution = calculateFinalCost(costMatrix, readSOL(str(path_to_file+fileName)+".sol"))

    #We calculate and print the final cost of our solution and the one of the optimised solution
    print("Total time :", endCVRP-startCVRP, "Clusturing time : ", ClusterTimer, "TSP time : ",TSPTimer)
    print("Quantum Resolution:", Quantum_Resolution)
    print("Optimal Resolution:", OptimalResolution)

    return (fileName, endCVRP-startCVRP, ClusterTimer, TSPTimer, Quantum_Resolution, OptimalResolution)
 











# -------------------------------------------------------------------------------------------- #
#                                         MAIN                                                 #
# -------------------------------------------------------------------------------------------- #

path_to_png = "/workspace/CVRPQuantumOptimization/PNG/"
path_to_file = "/workspace/CVRPQuantumOptimization/M/"

if not os.path.exists(path_to_png):
    os.makedirs(path_to_png)

#                              Prepare dataframe to stock everything
df = pd.DataFrame(columns = ['Name', 'TotalTime', 'ClusteringTime', 'TSPTime', 'QuantumResolutionScore', 'OptimalResolutionScore'])


#                                     Literature instances
# Literature instances from http://vrp.galgos.inf.puc-rio.br/index.php/en/ 

# Set A (Augerat, 1995) 

df = pd.DataFrame()

for f in os.listdir(path_to_file):
    if f[-3:] == "vrp":
        file = f[:-4]
        tuple_value = literatureGeneration(path_to_file, file, path_to_png)
        df = pd.concat((df, pd.DataFrame([{'Name' : tuple_value[0], 'TotalTime' : tuple_value[1], 'ClusteringTime' : tuple_value[2], 'TSPTime' : tuple_value[3], 'QuantumResolutionScore' : tuple_value[4], 'OptimalResolutionScore' : tuple_value[5]}])), axis = 0)
        df.to_csv("Result_set_M.csv", index = False,encoding='utf-8', sep = ";")

#tuple_value = literatureGeneration(path_to_file, "B-n31-k5", path_to_png)
#df = pd.concat((df, pd.DataFrame([{'Name' : tuple_value[0], 'TotalTime' : tuple_value[1], 'ClusteringTime' : tuple_value[2], 'TSPTime' : tuple_value[3], 'QuantumResolutionScore' : tuple_value[4], 'OptimalResolutionScore' : tuple_value[5]}])), axis = 0)
#tuple_value = literatureGeneration(path_to_file, "B-n34-k5", path_to_png)
#df = pd.concat((df, pd.DataFrame([{'Name' : tuple_value[0], 'TotalTime' : tuple_value[1], 'ClusteringTime' : tuple_value[2], 'TSPTime' : tuple_value[3], 'QuantumResolutionScore' : tuple_value[4], 'OptimalResolutionScore' : tuple_value[5]}])), axis = 0)

# Set E (Christofides and Eilon, 1969) 
# literatureGeneration("E-n30-k3")
# literatureGeneration("E-n101-k14")


# Set M (Christofides, Mingozzi and Toth, 1979) 
# literatureGeneration("M-n101-k10")
# literatureGeneration("M-n200-k17")




# CODE TO UPDATE FOR NEGATIVE COORDs
# Set F (Fisher, 1994)   
# literatureGeneration("F-n45-k4")
# Golden et al. (1998)
# literatureGeneration("Golden_1")
# Li et al. (2005) 
# literatureGeneration("Li_21")
# DIMACS (2021)
# literatureGeneration("Loggi-n401-k23")
# Limits of the machine ? To ckeck:
# Set X (Uchoa et al. (2014))
#

"""
#                                       self generation
numberOfVehicles        = 3
numberOfCities      = 20
capaConsumptionMin  = 1
capaConsumptionMax  = 4
selfgeneration(numberOfVehicles, numberOfCities, capaConsumptionMin, capaConsumptionMax)
"""
