
# Bank Failure Prediction 

This is my solution to the homework 2 of 95851 Data Science for Product Managers. I believe it is a good chance to apply easy classification model to predict bank failure. 

Dataset provides a time series snapshot for 396 banks status, with indicators of financial health.We care about what factors lead to the failure and can we predict the failfure in the future?

## Import Dataset


```python
# import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

bank = pd.read_excel('Bank failure data.xlsx')

bank.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4060 entries, 0 to 4059
    Data columns (total 14 columns):
    Bank Name                         4060 non-null object
    Quarter                           4060 non-null object
    Tier One                          4060 non-null float64
    Texas                             3997 non-null float64
    Size                              4060 non-null float64
    Brokered Deposits                 4040 non-null float64
    Net Chargeoffs                    4054 non-null float64
    Constr and Land Dev Loans         4060 non-null float64
    Change in Portfolio Mix           4060 non-null float64
    NP CRE to Assets                  4060 non-null float64
    Volatile Liabilities to Assets    4060 non-null float64
    Securities                        4060 non-null float64
    Failed during 2010Q2              4060 non-null object
    Cert Number                       4060 non-null int64
    dtypes: float64(10), int64(1), object(3)
    memory usage: 444.1+ KB
    


```python
bank.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bank Name</th>
      <th>Quarter</th>
      <th>Tier One</th>
      <th>Texas</th>
      <th>Size</th>
      <th>Brokered Deposits</th>
      <th>Net Chargeoffs</th>
      <th>Constr and Land Dev Loans</th>
      <th>Change in Portfolio Mix</th>
      <th>NP CRE to Assets</th>
      <th>Volatile Liabilities to Assets</th>
      <th>Securities</th>
      <th>Failed during 2010Q2</th>
      <th>Cert Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Exchange Bank</td>
      <td>2007Q4</td>
      <td>14.90</td>
      <td>19.36</td>
      <td>32.852108</td>
      <td>0.0</td>
      <td>0.03</td>
      <td>23.13</td>
      <td>3.38</td>
      <td>0.190681</td>
      <td>20.16</td>
      <td>99.07</td>
      <td>No</td>
      <td>160</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Exchange Bank</td>
      <td>2008Q1</td>
      <td>14.30</td>
      <td>20.86</td>
      <td>33.542390</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>32.96</td>
      <td>4.96</td>
      <td>0.000000</td>
      <td>21.23</td>
      <td>99.45</td>
      <td>No</td>
      <td>160</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Exchange Bank</td>
      <td>2008Q2</td>
      <td>14.15</td>
      <td>20.89</td>
      <td>34.140007</td>
      <td>0.0</td>
      <td>0.31</td>
      <td>33.71</td>
      <td>1.53</td>
      <td>0.022408</td>
      <td>19.69</td>
      <td>97.94</td>
      <td>No</td>
      <td>160</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Exchange Bank</td>
      <td>2008Q3</td>
      <td>14.13</td>
      <td>18.74</td>
      <td>34.038758</td>
      <td>0.0</td>
      <td>-0.02</td>
      <td>34.99</td>
      <td>3.80</td>
      <td>0.147452</td>
      <td>19.83</td>
      <td>98.84</td>
      <td>No</td>
      <td>160</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Exchange Bank</td>
      <td>2008Q4</td>
      <td>14.21</td>
      <td>21.82</td>
      <td>34.059328</td>
      <td>0.0</td>
      <td>1.21</td>
      <td>37.14</td>
      <td>3.86</td>
      <td>0.057306</td>
      <td>15.29</td>
      <td>99.84</td>
      <td>No</td>
      <td>160</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(bank['Bank Name'].unique())
```




    396



## What are the top two predictors of bank failure? 

First assume Whether a bank will collapse depends on its financial status on 2010Q1. We removed all the rows from the previous quarters. 388 bank info is collected and then the dataset is splitted into training dataset and test dataset. In this case, I choose decision tree classifier. To get the best hyperparameters, 10-fold cross validation is applied. Then the classifier is applied to entire training dataset and evaluted by the test dataset.

The top two predictors of bank failure is **Tier One** and **Change in Portfolio Mix**.


```python
Q1=bank[(bank['Quarter']=='2010Q1')]
Q1.drop('Cert Number',axis=1,inplace=True)
Q1.dropna(inplace=True)
```

#### There are 388 effective bank info at last.


```python
Q1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 388 entries, 9 to 4049
    Data columns (total 13 columns):
    Bank Name                         388 non-null object
    Quarter                           388 non-null object
    Tier One                          388 non-null float64
    Texas                             388 non-null float64
    Size                              388 non-null float64
    Brokered Deposits                 388 non-null float64
    Net Chargeoffs                    388 non-null float64
    Constr and Land Dev Loans         388 non-null float64
    Change in Portfolio Mix           388 non-null float64
    NP CRE to Assets                  388 non-null float64
    Volatile Liabilities to Assets    388 non-null float64
    Securities                        388 non-null float64
    Failed during 2010Q2              388 non-null object
    dtypes: float64(10), object(3)
    memory usage: 42.4+ KB
    

#### The dataset is a little inbalanced.


```python
Q1["Failed during 2010Q2"].value_counts()
```




    No     355
    Yes     33
    Name: Failed during 2010Q2, dtype: int64




```python
Q1=Q1.reset_index().drop('index',axis=1)
```

#### train-test split

Here, considering the fewer number of Positive ("Yes") cases, we split data in a stratified way. 

First, split data into training and test data. We run decision tree classifier and tune hyperparamters with 10-fold cv on training set and evaluate it on the test set.


```python
from sklearn.model_selection import train_test_split
X=Q1.drop(['Bank Name','Quarter','Failed during 2010Q2'],axis=1)
#X=X[X.columns[dt.feature_importances_>0]]
y=Q1['Failed during 2010Q2']

from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   stratify=y,
                                                    test_size=0.25,random_state=25)


```


```python
pd.Series(y_train).value_counts()
```




    No     266
    Yes     25
    Name: Failed during 2010Q2, dtype: int64




```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```


```python
X_train=X_train.reset_index().drop('index',axis=1)
y_train=y_train.reset_index().drop('index',axis=1)
```

#### Tune hyperparameters for decision tree classifiers

Considering the inbalanced dataset, we splited with stratfied methods.


```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10,random_state=25)
#skf.get_n_splits(X_train, y_train)
for maxd in range(1,2):
    accurate=[]
    for train_index, test_index in skf.split(X_train, y_train):
        X_Train, X_dev = X_train.iloc[train_index], X_train.iloc[test_index]
        y_Train, y_dev = y_train.iloc[train_index], y_train.iloc[test_index]
        dt = DecisionTreeClassifier(criterion="entropy",max_features=3,max_depth=2,random_state=25)
        dt.fit(X_Train,y_Train)
        pred=dt.predict(X_dev)
        accurate.append(accuracy_score(y_dev, pred))
    print(np.mean(accurate))
```

    0.986075533662
    

#### Train the model on training data and predict on test data.

Accuracy is not adequate for inbalanced data. So we print out the classification report and confusion matrix to see how it performs on both classes respectively. The evaluation is ok since only one bank was mislabelled as "yes". The accuracy is great and precision on minority class is acceptable. 


```python
dt=DecisionTreeClassifier(criterion="entropy",max_features=3,max_depth=2,random_state=25)
dt.fit(X_train,y_train)
print(classification_report(y_test,dt.predict(X_test)))
```

                 precision    recall  f1-score   support
    
             No       1.00      0.99      0.99        89
            Yes       0.89      1.00      0.94         8
    
    avg / total       0.99      0.99      0.99        97
    
    


```python
accuracy_score(y_test,dt.predict(X_test))
```




    0.98969072164948457




```python
confusion_matrix(y_test,dt.predict(X_test))
```




    array([[88,  1],
           [ 0,  8]], dtype=int64)



#### Find feature importance in decision tree model


```python
dt.feature_importances_
```




    array([ 0.98154267,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.01845733,  0.        ,  0.        ,  0.        ])




```python
X.columns[dt.feature_importances_>0]
```




    Index(['Tier One', 'Change in Portfolio Mix'], dtype='object')



##  Which banks are most likely to fail in the near future (and why)? 

A: Apply the model to predict the survival of bank which did survive during 2010Q2. The model predicts if a bank fails in 2010Q2. Alghough the bank did not fail at that time, it still has high risk of failure. Assume their financial status would not change a lot during time, then its failure is coming soon. Four banks were suggested collapse. They are State Central Bank, Gulf State Community Bank, ShoreBank, Darby Bank & Trust Company.



```python
potential1 = Q1[Q1['Failed during 2010Q2']=="No"].reset_index().drop('index',axis=1)
potential = potential1.drop(['Bank Name','Quarter','Failed during 2010Q2'],axis=1)
potential.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tier One</th>
      <th>Texas</th>
      <th>Size</th>
      <th>Brokered Deposits</th>
      <th>Net Chargeoffs</th>
      <th>Constr and Land Dev Loans</th>
      <th>Change in Portfolio Mix</th>
      <th>NP CRE to Assets</th>
      <th>Volatile Liabilities to Assets</th>
      <th>Securities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.25</td>
      <td>54.35</td>
      <td>34.285794</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>34.49</td>
      <td>1.53</td>
      <td>1.770006</td>
      <td>21.31</td>
      <td>102.34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.64</td>
      <td>21.32</td>
      <td>48.378092</td>
      <td>6.98</td>
      <td>0.61</td>
      <td>20.27</td>
      <td>2.66</td>
      <td>4.507692</td>
      <td>43.05</td>
      <td>100.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.01</td>
      <td>7.34</td>
      <td>13.818530</td>
      <td>0.00</td>
      <td>-0.06</td>
      <td>2.38</td>
      <td>3.22</td>
      <td>0.646950</td>
      <td>26.74</td>
      <td>103.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.00</td>
      <td>7.04</td>
      <td>48.322100</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.37</td>
      <td>3.93</td>
      <td>0.247313</td>
      <td>8.21</td>
      <td>100.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.51</td>
      <td>39.16</td>
      <td>73.719818</td>
      <td>19.35</td>
      <td>1.12</td>
      <td>22.74</td>
      <td>6.14</td>
      <td>1.664389</td>
      <td>11.14</td>
      <td>101.76</td>
    </tr>
  </tbody>
</table>
</div>



#### Find the probability of failure of banks

Find the banks whose predictions of failure is highly probable. 



```python
(pd.DataFrame(dt.predict_proba(potential),columns=["No","Yes"])["Yes"]>0.8).sort_values(ascending=False).head()
```




    63      True
    236     True
    167     True
    153     True
    354    False
    Name: Yes, dtype: bool




```python
potential1["Bank Name"][[63,236,167,153]]
```




    63             State Central Bank
    236     Gulf State Community Bank
    167                     ShoreBank
    153    Darby Bank & Trust Company
    Name: Bank Name, dtype: object



#### State Central Bank, Gulf State Community Bank(closed on Nov. 2010), ShoreBank(closed on Aug. 2010), Darby Bank & Trust Company(closed on Nov. 2010)

The prediction is pretty good. Adjusting the threshold, more banks will come out. 
