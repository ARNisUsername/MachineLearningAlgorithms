#Linear Regression:

from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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
