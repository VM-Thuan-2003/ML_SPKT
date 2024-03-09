from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, Y = load_iris(return_X_y=True)
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=100)

clf = MLPClassifier(alpha=10)

clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)

print(train_score, test_score)
