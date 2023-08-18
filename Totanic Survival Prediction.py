import pandas as pd
import numpy as np
df=pd.read_csv('c:users\hajia\desktop\one.csv')
df['Male']=df['Sex']=='male';
X=df[['Pclass','Male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y=df['Survived'].values;
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,y);
y_pred=model.predict(X);
number=(y==y_pred).sum();
print("Accuracy Score: ", number*100/887,"%");
#you can enter your own attributes instead of X in the form of:
#[[class, if male=True if female=False, Age, Siblings/Spouses,Parents/Children,Fare]]
#you will get [1] or [0] which is will survive or won't survive.
#for example:
print(model.predict([[3,False,72,0,1,12]]));
print(model.predict([[1,True,22,0,1,12]]));
