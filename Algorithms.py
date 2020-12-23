#Pandas locate based on index
import pandas as pd
df = pd.read_csv('[name].csv')
print(df.iloc[[12]]) #Locates everything in index 12

#Make mglearn wave(Not Machine Learning, just useful for future reference)
import mglearn
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#KNeighborsClassifier:

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

#Linear Regression:

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train, y_train)

#Ridge Regression(uses regularization: each feature should have as little affect on the outcome as possible while
#being able to predict correctly to avoid overfitting) - increase alpha means increase in generalization, decrease in training accuracy

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0).fit(X_train, y_train)

#Lasso(Alternate to Ridge, but some feautures become completely ignored to make the model easier to intepret; L1 Regularization)
#An increase in alpha would be an increase in amount of feautures used

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)

#Logistic Regression(One type of linear model used for classification; L2 Regularization by default)
#The higher C is, the increased complexity of the model(which makes the predictory better in most cases)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1,max_iter=10000,penalty="l2").fit(X_train, y_train)

#Decision Tree(greater the depth, the more overfitting)

from sklearn.tree import DecisionTreeClassifier

tr = DecisionTreeClassifier(max_depth=4,random_state=0).fit(X_train, y_train)

#Random Forest/RandomForestClassifier(the more n_estimators, the more trees in the voting process, You should have as much trees as possible to make the voting more accurate)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10,random_state=0).fit(X_train, y_train)

#Gradiatent Boosted Regression Trees(USED VERY OFTEN IN MACHINE LEARNING COMPETITIONS) Builds trees where each tree tries to correct the mistakes of the previous one
#Does not work well with high dimensional sparse data(data with a lot of elements and a lot of 0's)
#Learning rate: Higher means trees try stronger to correct other mistakes
#Max depth: Lower means lower compexity of the tree, low max_depth for gradient boosted models

from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(random_state=0,max_depth=3,learning_rate=0.1).fit(X_train, y_train)

#SVMs: A type of linear model which uses the kernel trick to transform your data and find the best boundary between all the possible feautures of outputs(POWERFUL!!!)
#Kernel trick: 'poly'/Polynomial Kernel: Computes all possible polynomials up to a certain degree of the original feautures(e.g. feauture1 ** 2 * feauture2 ** 5)
#Kernel trick: 'rbf'/Gaussian Kernel: It consideres all possible polynomials for all degrees, but the importance of each feauture decreases with a higher degree
#gamma: low gamma means lower complexity of model, high gamma means higher complexity of model
#C: low C means each data point has a limited influence on model, high C means each datapoint has higher influence on model(use higher C when model underfitting)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#scale data to 0 and 1
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
#calculate svm
svm = SVC(kernel='rbf',C=10,gamma=0.1).fit(X_train_scaled, y_train)

#Neural networks: Reduces the number of nodes([12,12] has 12 nodes) reduces complexity, increasing it increases the complexity
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10,10]).fit(X_train, y_train)

#Grid Search CV(Perfect for finding what parameters to put for your specific model!)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors':[1,2,3,4,5,6,7,8,9]}
grid = GridSearchCV(KNeighborsClassifier(),param_grid,verbose=3)
print(grid.best_params_)
model = grid.fit(X_train, y_train)

#K Means Cluserting 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = make_blobs(n_samples=150,n_features=2,centers=4,cluster_std=1.8,random_state=0) 
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(data[0])

fig, axes = plt.subplots(1,2)
axes[0].scatter(data[0][:,0],data[0][:,1],c=data[1])
axes[0].set_title('Original')
axes[1].scatter(data[0][:,0],data[0][:,1],c=model.labels_)
axes[1].set_title('K Means')
plt.show()

#MinMaxScaler(scales data, very common for neural networks
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Get embedding vectors of document
from gensim.models import Word2Vec
sentences = [["Hello there"], ["Hello my friend"], ["Goodbye friend"]]
model = Word2Vec(sentences, min_count=1)
vocab = list(model.wv.vocab)
vector_hello = model["Hello"]

#Save sklearn models
from sklearn.neighbors import KNeighborsClassifier
import joblib
model = KNeighborsClassifier().fit(X, y)
joblib.dump(model, 'neighbors_model.sav')
model = joblib.load('neighbors_model.sav')

#---------------TENSORFLOW NEURAL NETWORK--------
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

#Basic regression neural network
model = keras.Sequential()
model.add(keras.layers.Dense(11, activation='relu'))
model.add(keras.layers.Dense(11, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam',loss='mse')
stopping = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=10)
model.fit(X_train, y_train, epochs=5, batch_size=50, validation_loss=(X_test, y_test), callbacks=[stopping])
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

#Basic classification neural network
model = keras.Sequential()
model.add(keras.layers.Dense(11, activation='relu'))
model.add(keras.layers.Dense(11, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

#Basic LSTM neural network
model = keras.Sequential()
model.add(keras.layers.LSTM(128,input_shape=(28,28),return_sequences=True))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10,activation='softmax'))
opt = keras.optimizers.Adam(lr=1e-3,decay=1e-5)
model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=100)

#Save a neural network
model = keras.Sequential()
model.add(keras.layers.Dense(2,activation='sigmoid'))
model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=100)
model.save_weights("my_good_model")

same_model = keras.Sequential()
same_model.add(keras.layers.Dense(2,activation='sigmoid'))
same_model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
same_model.load_weights("my_good_model")

#TEXT CLASSIFICATION: Vectorize with sklearn and use Keras
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
vect = CountVectorizer().fit(X)
def transform_text(text):
    text = [text]
    return list(vect.transform(text).toarray())[0]
X = np.array([transform_text(text) for text in X])
y = np.array(y)

max_index = len(vectorizer.vocabulary_)
the_len = len(transform_text("Test text"))
dims = 16
global_pooling = True

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=max_index, output_dim=dims,
                                 input_length=the_len))
model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                              input_shape=(dims, the_len)))
model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                              input_shape=(dims, the_len)))
model.add(keras.layers.Dropout(0.2))
if global_pooling:
  model.add(keras.layers.GlobalAveragePooling1D())
else:
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y, epochs=1, batch_size=64)

#Use tf.GradientTape() for training
import tensorflow as tf
import numpy as np

#We will create a 2600 dataset, with the numbers below 1 being correlated to 0, while the numbers above 2 being correlated with 1
AMOUNT = 1300
X_1 = list(np.random.rand(AMOUNT,1))
X_2 = list(np.random.rand(AMOUNT,1)+2)
X = np.array(X_1 + X_2)
y = np.array([0 for i in range(AMOUNT)] + [1 for i in range(AMOUNT)])

#Here we will make our model. In this case, this will be a simple 10x1 neural network
def make_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

#Our model loss will be Mean Squared Error, and we will create a function to calculate that
def model_loss(y_true, y_pred):
  assert len(y_true) == len(y_pred)
  return sum([(y_true[i] - y_pred[i])**2 for i in range(len(y_true))])/len(y_true)

#We will create the model with our function, and set the optimizer to a learning rate of 0.1 as it is a small dataset
model = make_model()
model_optimizer = tf.keras.optimizers.Adam(0.1)
#This will be the function taken for a single epoch
def train_step(X, y):
  #We will use tf.GradientTape and calculate the loss while storing the neccessary info in the tape
  with tf.GradientTape() as tape:
      predictions = list(model(X))
      loss = model_loss(y, predictions)
  #We will now use the derivatives of our loss function to calculate the gradient between the loss and the model parameters
  gradients = tape.gradient(loss, model.trainable_variables)
  #Using the optimizer, 'gradients' will have the updated value for each variable, and we will apply them to the optimizer in a 0.1 learning rate
  model_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
#This is the actual train function, which does a train step per epoch
def train(X, y, epochs):
    for i in range(epochs):
        train_step(X, y)

#This function gives a list of predictions based on a list of X_pred
def make_prediction(X_pred):
    X_pred = np.array(X_pred).reshape(-1,1)
    return list(model.predict(X_pred))

#A function that can take a sigmoid input and output either 0 or 1
def do_round(num):
    if num >= 0.5:
        return 1
    return 0

#This is a function where, given amount_test, it will calculate the accuracy of the model 
def get_acc(amount_test):
  X_0 = np.random.rand(amount_test,1)
  X_1 = np.random.rand(amount_test,1)+2
  y_true = [0 for i in range(amount_test)] + [1 for i in range(amount_test)]
  y_pred = list(make_predictions(X_0)) + list(make_predictions(X_1))
  y_pred = [do_round(y_pred[i]) for i in range(len(y_pred))]
  amount_correct = [y_pred[i] == y_true[i] for i in range(len(y_pred))]
  return amount_correct.count(True)/len(y_pred)

#We will train our model on 7 epochs and get our accuracy, which should be close to 100%
train(X, y, 7)
print(get_acc(300))
print(make_predictions([0.1,0.2,0.3,0.5,0.9,1.2,1.3,1.8]))
