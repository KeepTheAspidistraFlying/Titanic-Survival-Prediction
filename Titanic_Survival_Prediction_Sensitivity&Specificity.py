import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
sensitivity_score=recall_score;
#sensivity & recall are the same & sensitivity score is not defined for sklearn, so we use recall.
from sklearn.metrics import precision_recall_fscore_support
#specifity is the recall of negative class.
def specificity_score(y_true,y_pred):
    p, r, f,s=precision_recall_fscore_support(y_true,y_pred)
    return r[0]
#"r" is "recall" and its first value is the recall of negative class with is specifity.
from sklearn.model_selection import train_test_split
df=pd.read_csv('C:Users\hajia\Desktop\one.csv');
df['male']=df['Sex']=='male';
X=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values;
y=df['Survived'].values;
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=5);
model=LogisticRegression();
model.fit(X_train,y_train);
y_pred=model.predict(X_test);

print(sensitivity_score(y_test,y_pred));
print(specificity_score(y_test,y_pred));