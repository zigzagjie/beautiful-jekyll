---
layout: post
title: Exploratory Data Analysis with Titanic Project Part I
subtitle: This is a introductory data science project walking through Python
gh-repo: zigzagjie/datascience
gh-badge: [star, fork, follow]
tags: [Python]
---

## Titanic Dataset is a classic Dataset for classfication problem in data science Competition platform: Kaggle. Through this project, I will walk through the entire data science pipeline, data collection, data manipulation, data wraggling, data visulization, data modeling and data evaluation. Machine learning techniques include logistic regression, SVM, decisioni trees, random forests and neural networks.

## This is the first part of the whole project. The codes is run on Python. 

### import important packages


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

### Read titanic traning dataset


```python
titanic = pd.read_csv('train.csv')
```

### Check the first 5 rows of train dataset


```python
titanic.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### Check the attributes we got


```python
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    

**Variable** | **Definition** | **Key**
------------ | -------------- | ---------------
survival  | Survival   | 0 = No, 1 = Yes
pclass  | Ticket class  |       1 = 1st, 2 = 2nd, 3 = 3rd
sex     |             Sex
Age      |         Age in years
sibsp     |        # of siblings / spouses aboard the Titanic
parch      |       # of parents / children aboard the Titanic
ticket      |      Ticket number
fare         |     Passenger fare
cabin         |    Cabin number
embarked       |   Port of Embarkation           |                     C = Cherbourg, Q = Queenstown, S = Southampton

attributes with its meaning

### Get the descriptive statistics of each column (only for numberic variables)


```python
titanic.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



There are some missing values in Age column

### Visualize some columns

#### **Visualize the distributions of Survivial**


```python
survive = titanic['Survived']
```


```python
survive.value_counts()
```




    0    549
    1    342
    Name: Survived, dtype: int64



More people did not survive


```python
sns.countplot(x='Survived',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2341656f278>




![png](/_posts/Titanic_Project/output_19_1.png)


#### Visualize the distribution of sex


```python
titanic['Sex'].value_counts()
```




    male      577
    female    314
    Name: Sex, dtype: int64




```python
sns.countplot(x='Sex',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23416696e80>




![png](/_posts/Titanic_Project/output_22_1.png)


#### Visualize the distribution of Age


```python
#sns.distplot(titanic['Age'])                  #error because of missing values
```

#### Drop null values in age


```python
sns.distplot(titanic['Age'].dropna())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x234168005c0>




![png](/_posts/Titanic_Project/output_26_1.png)


#### Use Pandas built-in visualization


```python
titanic['Age'].hist(bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23418c44668>




![png](/_posts/Titanic_Project/output_28_1.png)



```python
titanic['Age'].plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23418b92438>




![png](/_posts/Titanic_Project/output_29_1.png)


### Visualize the distribution of PClass


```python
titanic['Pclass'].value_counts()
```




    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64



### Visualize the distribution of Parch # parents/children


```python
sns.countplot(x='Pclass',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2341eb8aa90>




![png](/_posts/Titanic_Project/output_33_1.png)


### Visualize fares


```python
plt.figure(figsize=(10,3))
titanic['Fare'].hist(bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2341ece6b00>




![png](/_posts/Titanic_Project/output_35_1.png)



```python
sns.distplot(titanic['Fare'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2341edcf630>




![png](/_posts/Titanic_Project/output_36_1.png)


There might be some outliers in the Fare Column

## Visualize the distribution of #of Siblings/Spouses


```python
sns.countplot(x='SibSp',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23420106ba8>




![png](/_posts/Titanic_Project/output_39_1.png)



```python
sns.countplot(x='Parch',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23420112fd0>




![png](/_posts/Titanic_Project/output_40_1.png)


## Bi-variate Analysis

### Sex with Survival


```python
sns.countplot(x='Survived',hue='Sex',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2341e085f28>




![png](/_posts/Titanic_Project/output_43_1.png)


### Fare with Survival 


```python
sns.boxplot(x='Survived',y='Fare',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x234201d4668>




![png](/_posts/Titanic_Project/output_45_1.png)


### It indicated that people those survived had more expensive fares. Also, in People who survived, there was one cost over 500. It might indicate the outliers. 

### You can also visualize how sex differs with different fare and different sexes 


```python
sns.boxplot(x='Survived',y='Fare',hue='Sex',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2341ee50f98>




![png](/_posts/Titanic_Project/output_48_1.png)


### We may have to handle fares data

### Number of Sibilings/family 


```python
sns.countplot(hue='Survived',x='SibSp',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2341ee50e48>




![png](/_posts/Titanic_Project/output_51_1.png)


### People who have one sibling/spouse survived more likely that the single people


```python
sns.countplot(hue='Survived',x='Parch',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x234200ac2b0>




![png](/_posts/Titanic_Project/output_53_1.png)


## Missing Values

### there is missing value in age column and cabin column


```python
titanic.isnull().head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Visualize the missing values


```python
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b0d47885c0>




![png](/_posts/Titanic_Project/output_58_1.png)


### Cabin has a lot of missing values, we may just drop this colum. There are some missing values in Age column. We may have to handle them well because age might be an important attribute

### Next Part will introduce Handling Missing Values and Outliers
