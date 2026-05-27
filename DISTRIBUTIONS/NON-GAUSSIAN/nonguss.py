import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import PowerTransformer
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
Y_pred1 = clf2.predict(x_test)

print("Accuracy score LR = ", accuracy_score(y_test, Y_pred))
print("Accuracy score DT = ", accuracy_score(y_test, Y_pred1))

trf = FunctionTransformer(func = np.log1p)
x_train_transformed = trf.fit_transform(x_train)
x_test_transformed = trf.transform(x_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(x_train_transformed, y_train)
clf2.fit(x_train_transformed, y_train)

y_pred = clf.predict(x_test_transformed)
Y_pred1 = clf2.predict(x_test_transformed)

print("Accuracy LR", accuracy_score(y_test, y_pred))
print("Accuracy DT", accuracy_score(y_test, Y_pred1))

x_transformed = trf.fit_transform(x)
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

print("LR ", np.mean(cross_val_score(clf, x_transformed, y, scoring='accuracy', cv = 10)))
print("DT ", np.mean(cross_val_score(clf2, x_transformed, y, scoring='accuracy', cv= 10)))


plt.figure(figsize=(14,4))

plt.subplot(121)
stats.probplot(x_train['Fare'], dist="norm", plot=plt)
plt.title('Fare before log')

plt.subplot(122)
stats.probplot(x_train_transformed['Fare'], dist="norm", plot=plt)
plt.title("Fare after log")
plt.show()

plt.figure(figsize=(14,4))
plt.subplot(121)
stats.probplot(x_train['Age'], dist="norm", plot=plt)
plt.title("Age before log")

plt.subplot(122)
stats.probplot(x_train_transformed['Age'], dist="norm", plot=plt)
plt.title("Age after log")
plt.show()

trf2 = ColumnTransformer([('log', FunctionTransformer(np.log1p),['Fare'])], remainder='passthrough')

x_train_transformed2 = trf2.fit_transform(x_train)
x_test_transformed2 = trf2.transform(x_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(x_train_transformed2, y_train)
clf2.fit(x_train_transformed2, y_train)

y_pred = clf.predict(x_test_transformed2)
y_pred2 = clf2.predict(x_test_transformed2)

print("Accuracy LR", accuracy_score(y_test, y_pred))
print("Accuracy DT", accuracy_score(y_test, y_pred2))

x_transformed2 = trf2.fit_transform(x)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

print("LR ", np.mean(cross_val_score(clf, x_transformed2, y, scoring='accuracy', cv = 10)))
print("DT ", np.mean(cross_val_score(clf2, x_transformed2, y, scoring='accuracy', cv= 10)))


def apply_transform(transform):
    x = df.iloc[:,1:3]
    y = df.iloc[:,0]

    trf = ColumnTransformer([('log', FunctionTransformer(transform), ['Fare'])], remainder='passthrough')

    x_trans = trf.fit_transform(x)
    clf = LogisticRegression()

    print("Accuracy ", np.mean(cross_val_score(clf, x_trans, y, scoring='accuracy', cv=10)))

    plt.figure(figsize=(14,4))

    plt.subplot(121)
    stats.probplot(x['Fare'], dist="norm", plot= plt)
    plt.title("Fare Before Transform")

    plt.subplot(122)
    stats.probplot(x_trans[:,0], dist="norm", plot=plt)
    plt.title("Fare After Transform")

    plt.show()

    apply_transform(lambda x: x**2)