import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

df = pd.read_csv(r'C:\Sts ML\STATSTISTICS AND MACHINE LEARNINGS\DISTRIBUTIONS\NON-GAUSSIAN\train.csv', usecols=['Age', 'Fare', 'Survived'])

df['Age'] = df['Age'].fillna(df['Age'].mean())

x = df[['Age', 'Fare']]
y = df['Survived']

x_train, x_test, y_train, y_test  = train_test_split(x,y, test_size=0.2, random_state=42)
print(df.isnull().sum())

plt.figure(figsize=(14,4))
plt.subplot(121)
sns.histplot(x_train['Age'], kde=True)
plt.title('Age PDF')

plt.subplot(122)
stats.probplot(x_train['Age'], dist="norm", plot=plt)
plt.title('Age QQ plot')

# plt.show()

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(x_train, y_train)
clf2.fit(x_train, y_train)

Y_pred = clf.predict(x_test)
Y_pred2 = clf2.predict(x_test)

print("Accuracy score LR = ", accuracy_score(y_test, Y_pred))
print("Accuracy score DT = ", accuracy_score(y_test, Y_pred2))