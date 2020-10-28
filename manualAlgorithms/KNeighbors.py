import numpy as np

def convertDif(array, way='str'):
    for i in range(len(array)):
        if way == 'str':
            array[i] = str(array[i])
        elif way == 'int':
            array[i] = int(array[i])
    return array

def orderKeys(theDict):
    newDict = {}
    theKeys = np.sort(list(theDict.keys()))
    for key in theKeys:
        newDict[key] = theDict[key]
    return newDict

def countInstances(theList):
    theIn = list(np.unique(theList))
    theDict = {}
    for i in theIn:
        theDict[i] = 0
    for i in theList:
        theDict[i] += 1
    return theDict




#print(convertDif(list(newKey.split()),'int'))

def output_dict(x_array, y_array):
    info = {}
    for i in range(len(x_array)):
        newKey = ' '.join(convertDif(x_array[i]))
        info[newKey] = y_array[i]
    return info

def closest(theDict,point,k):
    info = {}
    
    for i in range(len(theDict.keys())):
        pointList = convertDif(list(list(theDict.keys())[i].split()),'int')
        xDif = np.absolute(point[0] - pointList[0])
        yDif = np.absolute(point[1] - pointList[1])
        distance = np.sqrt(xDif**2+yDif**2)

        pointClass = theDict[list(theDict.keys())[i]]
        if distance in list(info.keys()):
            distance += np.random.rand()/10000
        info[distance] = pointClass

    orderedDict = orderKeys(info)
    orderedClasses = list(orderedDict.values())

    match = countInstances(orderedClasses[:k])
    listMatches = list(match.values())
    if listMatches.count(listMatches[0]) == len(listMatches) and len(listMatches) > 1:
        match = countInstances(orderedClasses[:k-1])

    theKey = ''
    currentMax = 0
    for key in match.keys():
        if match[key] > currentMax:
            currentMax = match[key]
            theKey = key
            
    return theKey
    
class ManualKNeighbors:
    def __init__(self, k, xarray=None, yarray=None):
        self.k = k
        self.xarray = xarray
        self.yarray = yarray
        
    def fit(self, x_array, y_array):
        self.xarray = x_array
        self.yarray = y_array

    def predict(self, newPoint):
        return closest(output_dict(self.xarray,self.yarray), newPoint, self.k)
    

X = [[2,5],[2,6],[3,5],[3,6],[4,3],[4,4],[5,3],[5,4],[6,1],[6,2],[7,1],[7,2]]
y = [1,1,1,1,0,0,0,0,2,2,2,2]

knn = ManualKNeighbors(k=3)
knn.fit(X, y)
print(knn.predict([1,7])) #Predicts 1, like how it should
