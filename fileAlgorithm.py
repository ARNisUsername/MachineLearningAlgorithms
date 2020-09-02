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

#Random Forest/RandomForestClassifier(the more n_estimators, the more trees in the voting process)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10,random_state=0).fit(X_train, y_train)
