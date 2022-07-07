# KnowledgeDiscoveryInDatabase
This project aims to develop a system for classification. To develop this system, Python language is 
used. The features in this system includesthe function to read dataset, pre-process, divide the dataset
into 80% training and 20% testing for independent variable(X) and dependent variable(Y), Decision
Tree Classifier, Prediction, Confusion matrix, Accuracy score (classification accuracy), Precision score, 
Recall score, F1 score, AUC, percentage of correct classification, percentage of wrong classification
and classification report. There are several libraries to be downloaded in order to perform 
classification in python (pandas, matplotlib, scikit-learn). As such, PyCharm is used as the IDE. Decision 
tree is being used to perform classification and the algorithm that is used in decision tree is CART.
*The data file (excel) has to be placed in the same location as the python file.
*For AUC, 1 is represented as tested_positve and 0 is represented as tested_negative as AUC is 
calculated in int values. This is done by pre-processing the string values of tested_positive and 
tested_negative into 1 and 0
