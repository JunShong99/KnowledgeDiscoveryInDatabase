import pandas as pd
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report


#pd.set_option("display.max_rows", None, "display.max_columns", None) #to print everything

dataSet= pd.read_csv('dataset_37_diabetes.csv')#read dataset

dataSet.head()
print(dataSet)
print()

dataSet['class'] = dataSet['class'].replace(['tested_positive', 'tested_negative'], [1, 0])#pre-process string to int AUC, (1=tested_positive, 0=tested_negative)
feature_cols = ['preg', 'plas', 'pres', 'skin','insu','mass','pedi','age']#another name for indepent var
X = dataSet[feature_cols] #Features
y = dataSet['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)#X independent,Y dependent, (80% trainning, 20% testing)
clf = DecisionTreeClassifier(max_depth=3) #decisiontree

clf.fit(X_train, y_train)#fit data
y_pred = clf.predict(X_test)#start prediction
print(y_pred)#records,show prediction
print()

plt.figure(figsize = (15, 8))
plot_tree(clf, feature_names = feature_cols, class_names = ['tested_positive', 'tested_negative'], filled = True, rounded = True)


print(X_train)#80% trainning for X(independent variables)
print()

print(X_test)#20% testing for X(independent variables)
print()

print(y_train)#80% trainning for Y(dependent variables/target)
print()

print(y_test)#20% testing for Y(dependent variables/target)
print()

conMat=confusion_matrix(y_test, y_pred)#make confusion matrix
accSco=accuracy_score(y_test,y_pred)#predict target #dependent #Classification Accuracy
preSco=precision_score(y_test,y_pred)#Precision
recSco=recall_score(y_test,y_pred)#Recall
f1Sco=f1_score(y_test,y_pred)#F1
auCurve = roc_auc_score(y_test, y_pred)#AUC



print()
print("confusion matric is")
print(conMat)#+str,cannot be rounded
print()

correctClassfication=85+36
wrongClassification=14+19
totalClassification=correctClassfication+wrongClassification

print("Percentage of correct classification is")
print(round((correctClassfication)/(totalClassification) *100,2))
print()

print("Percentage of wrong classification is")
print(round((wrongClassification)/(totalClassification) *100,2))


print()
print("accuracy score is "+str(round(accSco,2)))
print("precision score is",round(preSco,2))
print("recall score is",round(recSco,2))
print("fi score is",round(f1Sco,2))
print("area under curve is",round(auCurve,2))
print()


report = classification_report(y_test,y_pred,target_names=["tested positive","tested negative"])#classsification report
print(report)

plt.show()