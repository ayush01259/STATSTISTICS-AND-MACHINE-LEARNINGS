import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer



df = pd.read_csv(r'C:\Sts ML\STATSTISTICS AND MACHINE LEARNINGS\DISTRIBUTIONS\NON-GAUSSIAN\concrete.csv')
print(df.head())

print(df.shape)
print(df.isnull().sum())
print("\n \n", df.describe())

x = df.drop(colums = ['Strength'])
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#Applying Regression without any tranformation

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print(r2_score(y_test, y_pred))


#cross checking wiht cross val score
print("\n \n")

lr = LinearRegression()
print(np.mean(cross_val_score(lr,x,y, scoring='r2')))


#Plotting the displots without any transformation

for col in x_train.colums:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.displot(x_train[col])
    plt.title(col)

    plt.subplot(122)
    stats.probplot(x_train[col], dist="norm", plot=plt)
    plt.title(col)

    plt.show()


# Applying box-cox transform

pt = PowerTransformer(method='box-cox')

x_train_transformed = pt.fit_transform(x_train+0.000001)
x_test_transformed = pt.transform(x_test+0.000001)
pd.DataFrame({'cols': x_train.columns, 'box_cox_lambdas':pt.lambdas_})


#Applying linear regression on transformed data

lr = LinearRegression()
lr.fit(x_train_transformed, y_train)
y_pred2 = lr.predict(x_test_transformed)
print("\n \n",r2_score(y_test, y_pred2))

#using cross_val_score

pt = PowerTransformer(method='box-cox')
x_transformed = pt.fit_transform(x+0.000001)
lr = LinearRegression()
print(np.mean(cross_val_score(lr,x_transformed, y, scoring='r2')))

#Before and after comparison for box_cox plot

for col in x_train_transformed.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(x_train[col])
    plt.title(col)

    plt.subplot(122)
    sns.distplot(x_train_transformed[col])
    plt.title(col)

    plt.show()


# Apply yeo-johnsn transform
# 
pt1 = PowerTransformer()
x_train_transformed2 = pt1.fit_transform(x_train)    
x_test_transformed2 = pt1.transform(x_test)

lr = LinearRegression()
lr.fit(x_train_transformed2, y_train)

y_pred3 = lr.predict(x_test_transformed2)
print(r2_score(y_test, y_pred3))

pd.DataFrame({'cols': x_train.columns, 'Yeo_Johnson_lambdas': pt1.lambdas_})


# applying cross val score
pt = PowerTransformer()
x_transformed2 = pt.fit_transform(x)

lr = LinearRegression()
np.mean(cross_val_score(lr,x_transformed2, y, scoring='r2'))

x_train_transformed2 = pd.DataFrame(x_train_transformed2, columns=x_train.columns)

# before and after comparison for yeo-johnson

for col in x_train_transformed2.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.displot(x_train[col])
    plt.title(col)

    plt.subplot(122)
    sns.distplot(x_train_transformed2[col])
    plt.title(col)

    plt.show()

# side by side lambdas
pd.DataFrame({'cols':x_train.columns, 'box_cox_lambdas':pt.lambdas_, 'Yeo_Johnson_lambdas': pt1.lambdas_})