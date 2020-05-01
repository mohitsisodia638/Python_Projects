#Description: This program dstects breast cancer, based off of data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the data
df = pd.read_csv('data.csv')
df.head()

#Count the number of rows and columns in data
df.shape

#Count the number of empty (NaN,na) values in column
df.isna().sum()

#Drop columns with missing values
df = df.dropna(axis=1)
df.shape

#Get number of Malignant (M) or Bengin (B) cells malicious or non malicious
df['diagnosis'].value_counts()

#Visualise the count
sns.countplot(df['diagnosis'],label='count')

#Look at the datatypes to see which columns need to be encoded
df.dtypes

#Encode the catogarical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)

#create pair plot
sns.pairplot(df.iloc[:,1:5], hue='diagnosis')

#print first five rows
df.head()

#Get the correlation of the columns
df.iloc[:,1:12].corr()

#Visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr() ,annot=True , fmt='.0%')

#Split the datasets into independent (x) and dependent (y) datasets
X = df.iloc[:,2:31].values
y = df.iloc[:,1].values

#split the datsets in 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train ,y_test = train_test_split(X,y , test_size = 0.25 ,random_state =0)

#Scale the data (feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Create a function for model
def models(X_train,y_train):
    
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train , y_train)
    
    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
    tree.fit(X_train , y_train)
    
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10 , criterion='entropy',random_state=0)
    forest.fit(X_train , y_train)
    
    #Using KNeighborsClassifier 
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn.fit(X_train, Y_train)
    
    #Using SVC linear
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, Y_train)
    
    #Using SVC rbf
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, Y_train)
    
    #Using GaussianNB 
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)

    #Print the models accuracy on training data
    print('[0]Logistic Regression Accuracy',log.score(X_train,y_train))
    print('[1]Decision Tree Accuracy',tree.score(X_train,y_train))
    print('[2]Random Forests Accuracy',forest.score(X_train,y_train))
    print('[3]SVC Linear Accuracy',svc_lin.score(X_train,y_train))
    print('[4]SVC rbf Accuracy',svc_rbf.score(X_train,y_train))
    print('[5]Guassian Naive Bias Accuracy',gauss.score(X_train,y_train))
    
    
    return log, tree, forest,svc_lin,svc_rbf,gauss

#Getting all models
model = models(X_train,Y_train)

#test model accuracy on test data on confusion matrix
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
    print('Model',i)
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    print(cm)
    print('Testing Accuracy =',(TP+TN)/(TP+TN+FN+FP))
    print()

#Show another way to get matrics of the models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
    print('Model',i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print(accuracy_score(Y_test,model[i].predict(X_test)))
    print()

#print the prediction of random forest Classifier Model
pred =model[2].predict(X_test)
print(pred)
print()
print(Y_test)