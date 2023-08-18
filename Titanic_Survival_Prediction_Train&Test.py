import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df=pd.read_csv('C:Users\hajia\Desktop\one.csv');
df['male']=df['Sex']=='male';
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values;
y=df['Survived'].values;
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=54,train_size=78);
#u can use train_test_split(X,y).
#random state is actualy the seed.
#train size is the ratio of train set to the whole daraset.
model=LogisticRegression();
model.fit(X_train,y_train);
y_pred=model.predict(X_test);
# train sets are for training and test sets are for testing!!!!
print(accuracy_score(y_test,y_pred));