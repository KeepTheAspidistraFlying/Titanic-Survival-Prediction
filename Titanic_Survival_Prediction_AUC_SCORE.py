import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
sensitivity_score=recall_score;
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df=pd.read_csv('C:Users\hajia\Desktop\one.csv');
df['male']=df['Sex']=='male';
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values;
y=df['Survived'].values;
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=5);

model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred_proba1 = model1.predict_proba(X_test)
print("model 1 AUC score:", roc_auc_score(y_test, y_pred_proba1[:, 1]))

model2 = LogisticRegression()
model2.fit(X_train[:, 0:2], y_train)
y_pred_proba2 = model2.predict_proba(X_test[:, 0:2])
print("model 1 AUC score:", roc_auc_score(y_test, y_pred_proba2[:, 1]))