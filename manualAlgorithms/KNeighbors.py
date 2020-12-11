import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def distance_vector(vector1, vector2):
    distance_n = []
    for n in range(len(vector1)):
        distance_n.append((vector1[n]-vector2[n])**2)
    return np.sqrt(sum(distance_n))

class ManualKNeighbors:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        
    def fit(self, the_x, the_y):
        self.x = the_x
        self.y = the_y

    def predict(self, x_pred):
        all_distances = [distance_vector(x_pred, self.x[i]) for i in range(len(self.x))]
        return self.y[np.argmin(all_distances)]

    def predict_multiple(self, x_preds):
        return [self.predict(x_vector) for x_vector in x_preds]

    def accuracy(self, x_test, y_test):
        predictions = [self.predict(test_vector) for test_vector in x_test]
        amount_correct = [predictions[i] == y_test[i] for i in range(len(predictions))].count(True)
        return amount_correct/len(predictions)

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = ManualKNeighbors()
model.fit(X_train, y_train)
print(model.accuracy(X_test, y_test))
