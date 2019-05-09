import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
from sklearn.feature_selection import SelectFromModel
data= pd.read_csv(r"C:\Users\nizameddiny\PycharmProjects\student_exam\studentsperformance.csv")

print(data.head())
print(data.info())
print(data.shape)
print(data.describe())

## Well, we have non null features.
non_integers = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
integers = ["math score","reading score", "writing score"]

# Let's analyze numerical metrics
sns.pairplot(data)
plt.show()

#Let's analyze categorical metrics
for i in non_integers :
    for k in non_integers :
        if i!= k :
            sns.countplot(x=data[i],hue=data[k], data= data)
            plt.show()

## Draw a box plot for observe any extraordinary integer in data



for i in non_integers :
    for a in integers :
        data.boxplot(by=i,column=a,grid=False)
        plt.ylabel(a)
        plt.xlabel(i)
        plt.show()


for i in non_integers :
    for a in integers :
        plt.scatter(x=data[a],y=data[i])
        plt.show()

## Let's look at this features  in different ways.

# Before going correlation, turn categorical values into dummies.


for i in non_integers :
    data[i] = pd.get_dummies(data[i])

# Now lets start with correlation. Better if we mask the heatmap due to symmetry.

data_corr = data.corr()

f, ax = plt.subplots(figsize=(9, 8))
mask = np.zeros_like(data_corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(data_corr, ax=ax, cmap="YlGnBu", linewidths=0.1,annot=True,mask=mask)
    plt.show()

# Second one is Cramer's V rule.
# Third one is Theil's U.

# Lets create average score as (math score + reading score + writing score) / 3

data["Average Score"] = (data["math score"] + data["reading score"] + data["writing score"])/3

# We cluster the student population to 4 groups
# But first we need to normalize Average Scores to improve our algoritms.

print(data["Average Score"].describe())

# Standardize the data before clustering. We use standardization because we have outliers a lot.

sc = StandardScaler()
data["Average Score"] = sc.fit_transform(data["Average Score"].values.reshape(-1,1))


# list for determining the n_estimators an explained variance
exp_var =[]
estimators = []

for i in range(10,1000,10) :
    X_train, X_test, y_train , y_test = train_test_split(data.drop(["Average Score", "math score" , "writing score" , "reading score"],axis=1), data["Average Score"],test_size=0.2)
    clf = RandomForestRegressor(n_estimators=i , random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    exp_var.append(explained_variance_score(y_test,y_pred))
    estimators.append(i)

# draw explained variance by number of estimators

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(estimators,exp_var,marker='o',
       linewidth=4,markersize=12)
ax.set_ylim([-1,1])
ax.set_xlim([10,1000])
plt.title('Expected Variance of Random Forest Regressor')
plt.ylabel('Expected Variance')
plt.xlabel('Trees in Forest')
plt.grid()
plt.show()

#rerun the model with best n_estimators
new_estimators = estimators[exp_var.index(min(exp_var))]
X_train, X_test, y_train , y_test = train_test_split(data.drop(["Average Score", "math score" , "writing score" , "reading score"],axis=1), data["Average Score"],test_size=0.2)
clf = RandomForestRegressor(n_estimators= new_estimators, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


## Regression Performance Evaluation


print(clf.score(X_test,y_test))
# Best possible score is 1.0, lower values are worse.
print(mean_absolute_error(y_test,y_pred))
# MAE output is non-negative floating point. The best value is 0.0.
print(mean_squared_error(y_test,y_pred))
# A non-negative floating point value (the best value is 0.0), or an array of floating point values, one for each individual target.
print(median_absolute_error(y_test,y_pred))
# A positive floating point value (the best value is 0.0).
print(r2_score(y_test,y_pred))
# R^2 (coefficient of determination) regression score function
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.


# Result is totally disaster. What if we classify the average score as percentile
# Let's design "Average Column Again"
data["Average Score"] = (data["math score"] + data["reading score"] + data["writing score"])/3

#There is no need to standardization for Random Forrest... decision trees are usually robust to numerical instabilities that sometimes impair convergence and precision in other algorithms.
#qcut chose by frequency, cut chose by space , its better the use qcut

data["Average Score"]=pd.qcut(data["Average Score"],2,labels=False)


# RandomForrest Begins...

# Let's create empty list for roc values to select best n_estimators

roc_list =[]
nest_list =[]

for i in range(50,2000,10) :

    X_train,X_test,y_train,y_test = train_test_split(data.drop(["Average Score", "math score" , "writing score" , "reading score"],axis=1),data["Average Score"],test_size=0.3)
    clf = RandomForestClassifier(n_estimators=500, random_state=7)
    clf.fit(X_train,y_train)
    roc=roc_auc_score(y_test,clf.predict(X_test))
    roc_list.append(roc)
    nest_list.append(i)

#Plotting the roc - i graph

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(nest_list,roc_list,marker='o',
       linewidth=4,markersize=12)
plt.title('Roc Score - Estimators')
plt.ylabel('Roc Score')
plt.xlabel('Estimators')
plt.grid()
plt.show()

# Select Best Estimator

i = nest_list[roc_list.index(min(roc_list))]

# Let's run RandomForest with best n_estimators parameter

clf = RandomForestClassifier(n_estimators=i, random_state=7)
clf.fit(X_train,y_train)
roc=roc_auc_score(y_test,clf.predict(X_test))
y_score = clf.predict_proba(X_test)[:,-1]
average_precision = average_precision_score(y_test, y_score)
precision, recall, _ = precision_recall_curve(y_test, y_score)
score=accuracy_score(y_test,clf.predict(X_test))
cr = classification_report(y_test, clf.predict(X_test))
print(score)
print(roc)
print(cr)
print("Precision: TP/(TP + FP) X Classı için Doğru Tahminlerim / X Clası için Toplam Tahminim ")
print("Recall : TP/(TP+FN) X Classı için Doğru Tahminlerim / Toplam X Classı Sayısı ")
print("f1-score: Harmonic mean of precision and recall ")
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

# Random Forrest Feature Importance

# We already defined our features as non_integers

for feature in zip(non_integers, clf.feature_importances_) :
    print(feature)

