import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
df = pd.read_csv(r"C:\Users\nizameddiny\PycharmProjects\Fraud Detection\creditcard.csv")
#Create an error list. We will use them later.
error = []
# Let's begin the explore our dataset.
print(df.shape)
print(df.describe())
# Our Features
print(df.head(3))
# What about the class numbers?
Fraud = df[df["Class"]== 1]
Normal = df[df["Class"]== 0]
Amount = df["Amount"]
print("The fraud to normal transaction ratio: " + str(Fraud.count()/Normal.count()))
plt.hist(Fraud["Amount"],color="Red",range=[0,600],bins=300)
plt.title("Fraud Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()
plt.hist(Normal["Amount"],range=[0,600],bins=300)
plt.title("Normal Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()
# The classes are so imbalanced which means we classify the data all as 1 we're going to have high accuracy.
# In order to solve this blocker issue, we may collect data to increase Class 2 number
# or we may select sample from our dataset randomly and get to work on! Second one is less costly.
# Lets look at our amount feature closer to find a way to resampling
print("Fraud amount statistics : \n"  + str(Fraud["Amount"].describe()))
print("Normal amount statistics : \n" + str(Normal["Amount"].describe()))
# Take data greater than 0 in order to make dataframe meaningful
rNormal = Normal[Normal["Amount"]>0]
rFraud  = Fraud[Fraud["Amount"]>0]
print(rNormal)
random_normal = rNormal.sample(frac=0.01)
new_df = pd.concat([rFraud,random_normal])
print(new_df)
# What we have about the problem :
# Need predict a category
# We have labeled data as Fraud and Normal
# We have less than 100K sample
# Lets start with the Linear SVC
X = new_df.iloc[:, 3:30]
y = new_df.iloc[:, 30]
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = svm.SVC(gamma="scale",kernel="linear")
clf.fit(X_train, y_train)
y_score = clf.decision_function(X_test)
average_precision = average_precision_score(y_test, y_score)
precision, recall, _ = precision_recall_curve(y_test, y_score)
score = accuracy_score(y_test,clf.predict(X_test))
roc = roc_auc_score(y_test,clf.predict(X_test))
cr = classification_report(y_test,clf.predict(X_test))
fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print("Class 1 is Fraud Class 0 is Normal Transactions")
print("Number of True Prediction / Number of Sample : " + str(score))
print("Precision: TP/(TP + FP) X Classı için Doğru Tahminlerim / X Clası için Toplam Tahminim ")
print("Recall : TP/(TP+FN) X Classı için Doğru Tahminlerim / Toplam X Classı Sayısı ")
print("f1-score: Harmonic mean of precision and recall ")
print(cr)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()
#KneighborsClassifier
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
i = error.index(min(error))
knn = KNeighborsClassifier(n_neighbors=i)
knn.fit(X_train, y_train)
pred_i = knn.predict(X_test)
print(confusion_matrix(y_test, pred_i))
print(classification_report(y_test, pred_i))

##Random forest
for i in range(100, 800, 50):
    clf=RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, -1]
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    score = accuracy_score(y_test,clf.predict(X_test))
    roc = roc_auc_score(y_test,clf.predict(X_test))
    cr = classification_report(y_test,clf.predict(X_test))
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("Class 1 is Fraud Class 0 is Normal Transactions")
    print("Number of True Prediction / Number of Sample : " + str(score))
    print("Precision: TP/(TP + FP) X Classı için Doğru Tahminlerim / X Clası için Toplam Tahminim ")
    print("Recall : TP/(TP+FN) X Classı için Doğru Tahminlerim / Toplam X Classı Sayısı ")
    print("f1-score: Harmonic mean of precision and recall ")
    print(cr)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
