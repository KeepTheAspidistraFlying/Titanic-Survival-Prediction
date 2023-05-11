# Titanic-Survival-Prediction
Using Machine Learning/ Logistic Regression

Hey Guyz;

so basically the code predicts whether or not someone with specific features survive Titanic. The accuracy score is a little more than 80%.

Ive used different packages in this code, such as numpy, pandas and scikitlearn.

Ive also written a few words to cover the basics of ML, if ure interested plz check "Basics of Machine Learning".

***************************************************************************************************************************

in this model the Precision(=TP/(TP+FP) is 0.7819 and Recall(=TP/TP+FN) is 0.6813, so the F1 score(=Harmonic mean of Precision and Recall) is 0.7281

code using Python:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score;

print(accuracy_score(y,y_pred));

print(precision_score(y,y_pred));

print(recall_score(y,y_pred));

print(f1_score(y,y_pred));

***************************************************************************************************************************

if u have any question or suggestion, plz contact me via my email; Hajiahmadiparisa@yahoo.com
