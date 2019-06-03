import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

data = pd.read_csv(r"C:\Users\nizameddiny\PycharmProjects\irisspecies\iris.csv")
print(data.describe())
print(data.columns)

# Feature Derivation

data["SepalArea"] = data["SepalLengthCm"]*data["SepalWidthCm"]
data["SepalHarmonic"] = data["SepalArea"]**(1/2)
data["PetalArea"] = data["PetalLengthCm"]*data["PetalWidthCm"]
data["PetalHarmonic"] = data["PetalArea"]**(1/2)

# id is meaningless. So we drop it down.

data = data.drop(["Id"],axis=1)

if data.isnull().any==True in data :
    print("There is null")
else :
    print("There is no null")



X_train, X_test , y_train , y_test = train_test_split(data.drop(["Species"],axis=1),data["Species"],test_size=0.5)

##SVM
print("SVM Results are as following")
SVM = svm.SVC(kernel="linear",gamma="scale")
SVM.fit(X_train,y_train)
y_pred1 = SVM.predict(X_test)
print(classification_report(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))

## Random Forest Classifier
print("RandomForrest Classifier Results are as following")
rfc = RandomForestClassifier(n_estimators=200, max_depth=4)
rfc.fit(X_train,y_train)
y_pred2 = rfc.predict(X_test)
print(classification_report(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))

# Neural Network
print("Neural Network Classifier Results are as following")

mlp = MLPClassifier(max_iter=500)
mlp.fit(X_train,y_train)
y_pred3 = mlp.predict(X_test)
print(classification_report(y_test,y_pred3))
print(accuracy_score(y_test,y_pred3))

# GradientBoosting Classifier

print("GradientBoosting Classifier Results are as following")

grd = GradientBoostingClassifier()
grd.fit(X_train,y_train)
y_pred4=grd.predict(X_test)
print(classification_report(y_test,y_pred4))
print(accuracy_score(y_test,y_pred4))

