---
layout: post
title: Data Preprocessing with Melbourne Housing Data
subtitle: Preprcessing is the key to the success
gh-repo: zigzagjie/datascience
gh-badge: [star, fork, follow]
tags: [Python]
---

# Data Preprocessing

## Introduction

In the real world, data is not clean. Several techniques are used as follows:

- Data cleaning: fill in missing values, smooth noisy data, identify or remove outliers, and resolve inconsistencies.
- Data integration: using multiple databases, data cubes, or files.
- Data transformation: normalization and aggregation.
- Data reduction: reducing the volume but producing the same or similar analytical results.
- Data discretization: part of data reduction, replacing numerical attributes with nominal ones.

## Case study 

The dataset I chose is from **Kaggle**. It is called Melbourne Housing Market. The following link is: https://www.kaggle.com/anthonypino/melbourne-housing-market

### 1. Import useful packages, read datasets and get some general information


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as ny
import seaborn as sns
```


```python
housing = pd.read_csv('housing.csv')
```


```python
housing.head()
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
      <th>Suburb</th>
      <th>Address</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>...</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbotsford</td>
      <td>68 Studley St</td>
      <td>2</td>
      <td>h</td>
      <td>NaN</td>
      <td>SS</td>
      <td>Jellis</td>
      <td>3/09/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>126.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.8014</td>
      <td>144.9958</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abbotsford</td>
      <td>85 Turner St</td>
      <td>2</td>
      <td>h</td>
      <td>1480000.0</td>
      <td>S</td>
      <td>Biggin</td>
      <td>3/12/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>202.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.7996</td>
      <td>144.9984</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abbotsford</td>
      <td>25 Bloomburg St</td>
      <td>2</td>
      <td>h</td>
      <td>1035000.0</td>
      <td>S</td>
      <td>Biggin</td>
      <td>4/02/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>79.0</td>
      <td>1900.0</td>
      <td>Yarra City Council</td>
      <td>-37.8079</td>
      <td>144.9934</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abbotsford</td>
      <td>18/659 Victoria St</td>
      <td>3</td>
      <td>u</td>
      <td>NaN</td>
      <td>VB</td>
      <td>Rounds</td>
      <td>4/02/2016</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yarra City Council</td>
      <td>-37.8114</td>
      <td>145.0116</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abbotsford</td>
      <td>5 Charles St</td>
      <td>3</td>
      <td>h</td>
      <td>1465000.0</td>
      <td>SP</td>
      <td>Biggin</td>
      <td>4/03/2017</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>150.0</td>
      <td>1900.0</td>
      <td>Yarra City Council</td>
      <td>-37.8093</td>
      <td>144.9944</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



The dataset had 21 attributes and 34857 observations in all.


```python
housing.isnull().sum()
```




    Suburb               0
    Address              0
    Rooms                0
    Type                 0
    Price             7610
    Method               0
    SellerG              0
    Date                 0
    Distance             1
    Postcode             1
    Bedroom2          8217
    Bathroom          8226
    Car               8728
    Landsize         11810
    BuildingArea     21115
    YearBuilt        19306
    CouncilArea          3
    Lattitude         7976
    Longtitude        7976
    Regionname           3
    Propertycount        3
    dtype: int64




```python
housing.isnull().sum()/len(housing)*100
```




    Suburb            0.000000
    Address           0.000000
    Rooms             0.000000
    Type              0.000000
    Price            21.832057
    Method            0.000000
    SellerG           0.000000
    Date              0.000000
    Distance          0.002869
    Postcode          0.002869
    Bedroom2         23.573457
    Bathroom         23.599277
    Car              25.039447
    Landsize         33.881286
    BuildingArea     60.576068
    YearBuilt        55.386293
    CouncilArea       0.008607
    Lattitude        22.882061
    Longtitude       22.882061
    Regionname        0.008607
    Propertycount     0.008607
    dtype: float64



It shows missing value percentage for each column. We can also get the percentage of all the missing values.


```python
missing = housing.isnull().sum().sum()
all = housing.isnull().count().sum()
missing/all*100
```




    13.794455441757275



There are a lot of missing values in the dataset, especially in Landsize, BuildingArea, and YearBuilt column. **13%** of the data are missing. That is not bad.

**You can also visualize missing values**


```python
plt.figure(figsize=(10,3))
sns.set(font_scale=1.2)
sns.heatmap(housing.isnull(),yticklabels = False, cbar = False, cmap = 'plasma')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x170625eb978>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_16_1.png)


### 2. Obtain descriptive statistics of some attributes.

Firstly, we need to understand the data type. Typically, they are:

1. Categorical data
  - Nominal: variables are variables that have two or more categories, but which do not have an intrinsic order.
  Examples: Gender, weather, room type
  - Ordinal: variables are variables that have two or more categories just like nominal variables only the categories can also be ordered or ranked.
  Examples: Class, Ranking
  
2. Numeric data
  - Interval
  - Ratio


Measures of the central tendency: Mean, Median, Mode

  - **Mean**: Average; Susceptible to outliers.
  - **Median**: Better in skewed data
  - **Mode**: the most frequent score. Best for nominal data
  - **Importance**: right-skewed data: median > mean. left-skewed data: median < mean 
  
  The more skewed the distribution, the greater the difference between the median and mean, and the greater emphasis should be placed on using the median as opposed to the mean.
  
  
  
Measures of the Dispersion: Variance/Standard Deviation, Range, interquantile range

  - **Variance/Standatd Deviation**
  - **Range**: Max-min, 
  - **Interquartile**: Q3 - Q1. Interquartile range is where the middle 50% data locates.


```python
housing.describe()
#'mean of housing price',housing['Price'].mean()
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
      <th>Rooms</th>
      <th>Price</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>34857.000000</td>
      <td>2.724700e+04</td>
      <td>34856.000000</td>
      <td>34856.000000</td>
      <td>26640.000000</td>
      <td>26631.000000</td>
      <td>26129.000000</td>
      <td>23047.000000</td>
      <td>13742.00000</td>
      <td>15551.000000</td>
      <td>26881.000000</td>
      <td>26881.000000</td>
      <td>34854.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.031012</td>
      <td>1.050173e+06</td>
      <td>11.184929</td>
      <td>3116.062859</td>
      <td>3.084647</td>
      <td>1.624798</td>
      <td>1.728845</td>
      <td>593.598993</td>
      <td>160.25640</td>
      <td>1965.289885</td>
      <td>-37.810634</td>
      <td>145.001851</td>
      <td>7572.888306</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.969933</td>
      <td>6.414671e+05</td>
      <td>6.788892</td>
      <td>109.023903</td>
      <td>0.980690</td>
      <td>0.724212</td>
      <td>1.010771</td>
      <td>3398.841946</td>
      <td>401.26706</td>
      <td>37.328178</td>
      <td>0.090279</td>
      <td>0.120169</td>
      <td>4428.090313</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>8.500000e+04</td>
      <td>0.000000</td>
      <td>3000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1196.000000</td>
      <td>-38.190430</td>
      <td>144.423790</td>
      <td>83.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>6.350000e+05</td>
      <td>6.400000</td>
      <td>3051.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>224.000000</td>
      <td>102.00000</td>
      <td>1940.000000</td>
      <td>-37.862950</td>
      <td>144.933500</td>
      <td>4385.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>8.700000e+05</td>
      <td>10.300000</td>
      <td>3103.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>521.000000</td>
      <td>136.00000</td>
      <td>1970.000000</td>
      <td>-37.807600</td>
      <td>145.007800</td>
      <td>6763.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>1.295000e+06</td>
      <td>14.000000</td>
      <td>3156.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>670.000000</td>
      <td>188.00000</td>
      <td>2000.000000</td>
      <td>-37.754100</td>
      <td>145.071900</td>
      <td>10412.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16.000000</td>
      <td>1.120000e+07</td>
      <td>48.100000</td>
      <td>3978.000000</td>
      <td>30.000000</td>
      <td>12.000000</td>
      <td>26.000000</td>
      <td>433014.000000</td>
      <td>44515.00000</td>
      <td>2106.000000</td>
      <td>-37.390200</td>
      <td>145.526350</td>
      <td>21650.000000</td>
    </tr>
  </tbody>
</table>
</div>



Here, we choose price and number of rooms as two attributes we are interested in.

Also, we may need **median, mode, range, quartiles** for these two attributes.


```python
'median of housing price',housing['Price'].median()
```

    median of housing price 870000.0
    




    ('quartiles for housing price', 870000.0)




```python
'mode of housing price',housing['Price'].mode()[0],housing['Price'].mode()[1]
```




    ('mode of housing price', 600000.0, 1100000.0)




```python
'mode of housing price',[x for x in housing['Price'].mode()]
```




    ('mode of housing price', [600000.0, 1100000.0])




```python
'range of housing price',housing['Price'].max()-housing['Price'].min()
```




    ('range of housing price', 11115000.0)




```python
'quartiles for housing price',housing['Price'].quantile([0.25,0.75])
```




    ('quartiles for housing price', 0.25     635000.0
     0.75    1295000.0
     Name: Price, dtype: float64)




```python
'standard deviation for housing price',housing['Price'].std()
```




    ('standard deviation for housing price', 641467.1301046001)



Visualize the distributions of numeric value

Histogram, Boxplot


```python
plt.figure(figsize=[15,6])
housing['Price'].hist(bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26f1caabfd0>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_27_1.png)



```python
plt.figure(figsize=[10,8])
sns.distplot(housing['Price'].dropna(),bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17062335128>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_28_1.png)



```python
sns.boxplot(y='Price',data=housing)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26f1ca8fe80>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_29_1.png)


Then apply the same techniques to **Rooms** column. 


```python
'median of room number',housing['Rooms'].median()
```




    ('median of room number', 3.0)




```python
'mode of room number',housing['Rooms'].mode()[0]
```




    ('mode of room number', 3)




```python
'range of room number',housing['Rooms'].max()-housing['Rooms'].min()
```




    ('range of room number', 15)




```python
'quartiles for room number',housing['Rooms'].quantile([0.25,0.75])
```




    ('quartiles for room number', 0.25    2.0
     0.75    4.0
     Name: Rooms, dtype: float64)




```python
plt.figure(figsize=[15,6])
housing['Rooms'].hist(bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26f1d1fc550>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_35_1.png)



```python
sns.countplot(housing['Rooms'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26f1d308c88>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_36_1.png)



```python
sns.boxplot(y=housing['Rooms'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26f1d316e48>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_37_1.png)


### 3. Potential problem of data quality

**1. Missing data**

   There are several methods to handle missing values. In summary, they are:
   
   - Ignore the tuple: usually done when class label is missing.
   - Use the attribute mean (or majority nominal value) to fill in the missing value.
   - Use the attribute mean (or majority nominal value) for all samples belonging to the same class.
   - Predict the missing value by using a learning algorithm: consider the attribute with the missing value as a dependent (class) variable and run a learning algorithm (usually Bayes or decision tree) to predict the missing value.
   
   Reference: http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html
   

**To simply, we just remove all the observations which have missing values. Advanced methods will be studied in the future.**


```python
housing = housing.dropna()
```


```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8887 entries, 2 to 34856
    Data columns (total 21 columns):
    Suburb           8887 non-null object
    Address          8887 non-null object
    Rooms            8887 non-null int64
    Type             8887 non-null object
    Price            8887 non-null float64
    Method           8887 non-null object
    SellerG          8887 non-null object
    Date             8887 non-null object
    Distance         8887 non-null float64
    Postcode         8887 non-null float64
    Bedroom2         8887 non-null float64
    Bathroom         8887 non-null float64
    Car              8887 non-null float64
    Landsize         8887 non-null float64
    BuildingArea     8887 non-null float64
    YearBuilt        8887 non-null float64
    CouncilArea      8887 non-null object
    Lattitude        8887 non-null float64
    Longtitude       8887 non-null float64
    Regionname       8887 non-null object
    Propertycount    8887 non-null float64
    dtypes: float64(12), int64(1), object(8)
    memory usage: 1.5+ MB
    

Now, we drop all the missing values. Dataset is clean. 


**2. Outlier data**
   
   There are several methods to handle missing values. In summary, they are:
   
   - Binning
       - Sort the attribute values and partition them into bins (see "Unsupervised discretization" below);
       - Then smooth by bin means,  bin median, or  bin boundaries.
   - Clustering: group values in clusters and then detect and remove outliers (automatic or manual) 
   - Regression: smooth by fitting the data into regression functions.
   - Univariate method: Boxplot


```python
housing.describe()
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
      <th>Rooms</th>
      <th>Price</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8887.000000</td>
      <td>8.887000e+03</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
      <td>8887.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.098909</td>
      <td>1.092902e+06</td>
      <td>11.199887</td>
      <td>3111.662653</td>
      <td>3.078204</td>
      <td>1.646450</td>
      <td>1.692247</td>
      <td>523.480365</td>
      <td>149.309477</td>
      <td>1965.753348</td>
      <td>-37.804501</td>
      <td>144.991393</td>
      <td>7475.940137</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.963786</td>
      <td>6.793819e+05</td>
      <td>6.813402</td>
      <td>112.614268</td>
      <td>0.966269</td>
      <td>0.721611</td>
      <td>0.975464</td>
      <td>1061.324228</td>
      <td>87.925580</td>
      <td>37.040876</td>
      <td>0.090549</td>
      <td>0.118919</td>
      <td>4375.024364</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.310000e+05</td>
      <td>0.000000</td>
      <td>3000.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1196.000000</td>
      <td>-38.174360</td>
      <td>144.423790</td>
      <td>249.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>6.410000e+05</td>
      <td>6.400000</td>
      <td>3044.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>212.000000</td>
      <td>100.000000</td>
      <td>1945.000000</td>
      <td>-37.858560</td>
      <td>144.920000</td>
      <td>4382.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>9.000000e+05</td>
      <td>10.200000</td>
      <td>3084.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>478.000000</td>
      <td>132.000000</td>
      <td>1970.000000</td>
      <td>-37.798700</td>
      <td>144.998500</td>
      <td>6567.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>1.345000e+06</td>
      <td>13.900000</td>
      <td>3150.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>652.000000</td>
      <td>180.000000</td>
      <td>2000.000000</td>
      <td>-37.748945</td>
      <td>145.064560</td>
      <td>10331.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12.000000</td>
      <td>9.000000e+06</td>
      <td>47.400000</td>
      <td>3977.000000</td>
      <td>12.000000</td>
      <td>9.000000</td>
      <td>10.000000</td>
      <td>42800.000000</td>
      <td>3112.000000</td>
      <td>2019.000000</td>
      <td>-37.407200</td>
      <td>145.526350</td>
      <td>21650.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing[housing['BuildingArea']==0]
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
      <th>Suburb</th>
      <th>Address</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>...</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7211</th>
      <td>North Melbourne</td>
      <td>19 Shands La</td>
      <td>2</td>
      <td>t</td>
      <td>841000.0</td>
      <td>S</td>
      <td>Jellis</td>
      <td>4/03/2017</td>
      <td>2.3</td>
      <td>3051.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>215.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>Melbourne City Council</td>
      <td>-37.79530</td>
      <td>144.94370</td>
      <td>Northern Metropolitan</td>
      <td>6821.0</td>
    </tr>
    <tr>
      <th>19775</th>
      <td>Balwyn North</td>
      <td>14 Wanbrow Av</td>
      <td>5</td>
      <td>h</td>
      <td>1950000.0</td>
      <td>S</td>
      <td>RT</td>
      <td>3/09/2017</td>
      <td>9.7</td>
      <td>3104.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>743.0</td>
      <td>0.0</td>
      <td>1949.0</td>
      <td>Boroondara City Council</td>
      <td>-37.80235</td>
      <td>145.09311</td>
      <td>Southern Metropolitan</td>
      <td>7809.0</td>
    </tr>
    <tr>
      <th>19840</th>
      <td>Bundoora</td>
      <td>22 Moreton Cr</td>
      <td>3</td>
      <td>h</td>
      <td>814000.0</td>
      <td>S</td>
      <td>Barry</td>
      <td>3/09/2017</td>
      <td>12.1</td>
      <td>3083.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>542.0</td>
      <td>0.0</td>
      <td>1970.0</td>
      <td>Banyule City Council</td>
      <td>-37.70861</td>
      <td>145.05691</td>
      <td>Northern Metropolitan</td>
      <td>10175.0</td>
    </tr>
    <tr>
      <th>20223</th>
      <td>Roxburgh Park</td>
      <td>16 Sandover Dr</td>
      <td>4</td>
      <td>h</td>
      <td>570000.0</td>
      <td>S</td>
      <td>Raine</td>
      <td>3/09/2017</td>
      <td>20.6</td>
      <td>3064.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>504.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>Hume City Council</td>
      <td>-37.61419</td>
      <td>144.93448</td>
      <td>Northern Metropolitan</td>
      <td>5833.0</td>
    </tr>
    <tr>
      <th>20262</th>
      <td>Thornbury</td>
      <td>19/337 Station St</td>
      <td>3</td>
      <td>t</td>
      <td>900000.0</td>
      <td>VB</td>
      <td>Jellis</td>
      <td>3/09/2017</td>
      <td>7.0</td>
      <td>3071.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>Darebin City Council</td>
      <td>-37.76343</td>
      <td>145.02096</td>
      <td>Northern Metropolitan</td>
      <td>8870.0</td>
    </tr>
    <tr>
      <th>22040</th>
      <td>Prahran</td>
      <td>6 Aberdeen Rd</td>
      <td>3</td>
      <td>h</td>
      <td>1390000.0</td>
      <td>S</td>
      <td>Marshall</td>
      <td>19/08/2017</td>
      <td>4.6</td>
      <td>3181.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>125.0</td>
      <td>0.0</td>
      <td>2002.0</td>
      <td>Stonnington City Council</td>
      <td>-37.85257</td>
      <td>145.00296</td>
      <td>Southern Metropolitan</td>
      <td>7717.0</td>
    </tr>
    <tr>
      <th>22507</th>
      <td>Huntingdale</td>
      <td>33 Beauford St</td>
      <td>3</td>
      <td>h</td>
      <td>1205000.0</td>
      <td>SA</td>
      <td>FN</td>
      <td>23/09/2017</td>
      <td>12.3</td>
      <td>3166.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>622.0</td>
      <td>0.0</td>
      <td>1960.0</td>
      <td>Monash City Council</td>
      <td>-37.90823</td>
      <td>145.10851</td>
      <td>Southern Metropolitan</td>
      <td>768.0</td>
    </tr>
    <tr>
      <th>22931</th>
      <td>Balwyn North</td>
      <td>1 Hosken St</td>
      <td>5</td>
      <td>h</td>
      <td>2800000.0</td>
      <td>S</td>
      <td>Marshall</td>
      <td>26/08/2017</td>
      <td>9.7</td>
      <td>3104.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1173.0</td>
      <td>0.0</td>
      <td>1960.0</td>
      <td>Boroondara City Council</td>
      <td>-37.80385</td>
      <td>145.09094</td>
      <td>Southern Metropolitan</td>
      <td>7809.0</td>
    </tr>
    <tr>
      <th>22994</th>
      <td>Brighton East</td>
      <td>60 Cummins Rd</td>
      <td>3</td>
      <td>h</td>
      <td>1650000.0</td>
      <td>SP</td>
      <td>Buxton</td>
      <td>26/08/2017</td>
      <td>10.3</td>
      <td>3187.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>623.0</td>
      <td>0.0</td>
      <td>1920.0</td>
      <td>Bayside City Council</td>
      <td>-37.92698</td>
      <td>145.02673</td>
      <td>Southern Metropolitan</td>
      <td>6938.0</td>
    </tr>
    <tr>
      <th>23022</th>
      <td>Bundoora</td>
      <td>37 Greenwood Dr</td>
      <td>4</td>
      <td>h</td>
      <td>815000.0</td>
      <td>S</td>
      <td>Ray</td>
      <td>26/08/2017</td>
      <td>12.1</td>
      <td>3083.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>525.0</td>
      <td>0.0</td>
      <td>1965.0</td>
      <td>Banyule City Council</td>
      <td>-37.70765</td>
      <td>145.05556</td>
      <td>Northern Metropolitan</td>
      <td>10175.0</td>
    </tr>
    <tr>
      <th>23085</th>
      <td>Craigieburn</td>
      <td>28 Powell St</td>
      <td>3</td>
      <td>h</td>
      <td>412500.0</td>
      <td>S</td>
      <td>RE</td>
      <td>26/08/2017</td>
      <td>20.6</td>
      <td>3064.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>197.0</td>
      <td>0.0</td>
      <td>2012.0</td>
      <td>Hume City Council</td>
      <td>-37.57687</td>
      <td>144.91100</td>
      <td>Northern Metropolitan</td>
      <td>15510.0</td>
    </tr>
    <tr>
      <th>23115</th>
      <td>Epping</td>
      <td>26 Lowalde Dr</td>
      <td>3</td>
      <td>h</td>
      <td>595000.0</td>
      <td>S</td>
      <td>hockingstuart</td>
      <td>26/08/2017</td>
      <td>19.6</td>
      <td>3076.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>536.0</td>
      <td>0.0</td>
      <td>1980.0</td>
      <td>Whittlesea City Council</td>
      <td>-37.64972</td>
      <td>145.04086</td>
      <td>Northern Metropolitan</td>
      <td>10926.0</td>
    </tr>
    <tr>
      <th>23159</th>
      <td>Glen Iris</td>
      <td>6 Viva St</td>
      <td>4</td>
      <td>h</td>
      <td>2690000.0</td>
      <td>PI</td>
      <td>Marshall</td>
      <td>26/08/2017</td>
      <td>7.3</td>
      <td>3146.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>647.0</td>
      <td>0.0</td>
      <td>1910.0</td>
      <td>Boroondara City Council</td>
      <td>-37.86133</td>
      <td>145.04167</td>
      <td>Southern Metropolitan</td>
      <td>10412.0</td>
    </tr>
    <tr>
      <th>23242</th>
      <td>Kew</td>
      <td>16 Hodgson St</td>
      <td>5</td>
      <td>h</td>
      <td>3450000.0</td>
      <td>PI</td>
      <td>Kay</td>
      <td>26/08/2017</td>
      <td>5.4</td>
      <td>3101.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>668.0</td>
      <td>0.0</td>
      <td>2006.0</td>
      <td>Boroondara City Council</td>
      <td>-37.80795</td>
      <td>145.01474</td>
      <td>Southern Metropolitan</td>
      <td>10331.0</td>
    </tr>
    <tr>
      <th>23250</th>
      <td>Kilsyth</td>
      <td>17 Birkenhead Dr</td>
      <td>3</td>
      <td>h</td>
      <td>803000.0</td>
      <td>S</td>
      <td>Max</td>
      <td>26/08/2017</td>
      <td>26.0</td>
      <td>3137.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>862.0</td>
      <td>0.0</td>
      <td>1970.0</td>
      <td>Maroondah City Council</td>
      <td>-37.79902</td>
      <td>145.32092</td>
      <td>Eastern Metropolitan</td>
      <td>4654.0</td>
    </tr>
    <tr>
      <th>23321</th>
      <td>Moorabbin</td>
      <td>7 Walsh Av</td>
      <td>3</td>
      <td>h</td>
      <td>1290000.0</td>
      <td>S</td>
      <td>Ray</td>
      <td>26/08/2017</td>
      <td>14.3</td>
      <td>3189.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>580.0</td>
      <td>0.0</td>
      <td>1970.0</td>
      <td>Kingston City Council</td>
      <td>-37.94492</td>
      <td>145.04938</td>
      <td>Southern Metropolitan</td>
      <td>2555.0</td>
    </tr>
    <tr>
      <th>23378</th>
      <td>Port Melbourne</td>
      <td>44 Garton St</td>
      <td>4</td>
      <td>t</td>
      <td>2455000.0</td>
      <td>SP</td>
      <td>Marshall</td>
      <td>26/08/2017</td>
      <td>3.5</td>
      <td>3207.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>123.0</td>
      <td>0.0</td>
      <td>2010.0</td>
      <td>Melbourne City Council</td>
      <td>-37.83349</td>
      <td>144.94840</td>
      <td>Southern Metropolitan</td>
      <td>8648.0</td>
    </tr>
    <tr>
      <th>23654</th>
      <td>Cheltenham</td>
      <td>5 Hannah St</td>
      <td>3</td>
      <td>h</td>
      <td>975000.0</td>
      <td>S</td>
      <td>O'Brien</td>
      <td>7/10/2017</td>
      <td>17.9</td>
      <td>3192.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>651.0</td>
      <td>0.0</td>
      <td>1970.0</td>
      <td>Bayside City Council</td>
      <td>-37.95683</td>
      <td>145.07184</td>
      <td>Southern Metropolitan</td>
      <td>9758.0</td>
    </tr>
    <tr>
      <th>23690</th>
      <td>Craigieburn</td>
      <td>18 Pymble Gdns</td>
      <td>4</td>
      <td>h</td>
      <td>590000.0</td>
      <td>S</td>
      <td>LJ</td>
      <td>7/10/2017</td>
      <td>20.6</td>
      <td>3064.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>448.0</td>
      <td>0.0</td>
      <td>2009.0</td>
      <td>Hume City Council</td>
      <td>-37.60902</td>
      <td>144.91279</td>
      <td>Northern Metropolitan</td>
      <td>15510.0</td>
    </tr>
    <tr>
      <th>24116</th>
      <td>Werribee</td>
      <td>21 Sinns Av</td>
      <td>3</td>
      <td>h</td>
      <td>550000.0</td>
      <td>S</td>
      <td>Ray</td>
      <td>7/10/2017</td>
      <td>14.7</td>
      <td>3030.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>580.0</td>
      <td>0.0</td>
      <td>1980.0</td>
      <td>Wyndham City Council</td>
      <td>-37.90136</td>
      <td>144.66925</td>
      <td>Western Metropolitan</td>
      <td>16166.0</td>
    </tr>
    <tr>
      <th>24196</th>
      <td>Balwyn North</td>
      <td>5 Highview Rd</td>
      <td>4</td>
      <td>h</td>
      <td>1500000.0</td>
      <td>VB</td>
      <td>Fletchers</td>
      <td>14/10/2017</td>
      <td>9.7</td>
      <td>3104.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>620.0</td>
      <td>0.0</td>
      <td>1965.0</td>
      <td>Boroondara City Council</td>
      <td>-37.78396</td>
      <td>145.07942</td>
      <td>Southern Metropolitan</td>
      <td>7809.0</td>
    </tr>
    <tr>
      <th>24205</th>
      <td>Bentleigh</td>
      <td>1 Donaldson St</td>
      <td>3</td>
      <td>h</td>
      <td>1730000.0</td>
      <td>S</td>
      <td>Jellis</td>
      <td>14/10/2017</td>
      <td>11.4</td>
      <td>3204.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>569.0</td>
      <td>0.0</td>
      <td>1940.0</td>
      <td>Glen Eira City Council</td>
      <td>-37.91456</td>
      <td>145.04109</td>
      <td>Southern Metropolitan</td>
      <td>6795.0</td>
    </tr>
    <tr>
      <th>24344</th>
      <td>Donvale</td>
      <td>29 Martha St</td>
      <td>2</td>
      <td>h</td>
      <td>1070000.0</td>
      <td>S</td>
      <td>hockingstuart</td>
      <td>14/10/2017</td>
      <td>16.1</td>
      <td>3111.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>758.0</td>
      <td>0.0</td>
      <td>1980.0</td>
      <td>Manningham City Council</td>
      <td>-37.79883</td>
      <td>145.17337</td>
      <td>Eastern Metropolitan</td>
      <td>4790.0</td>
    </tr>
    <tr>
      <th>24602</th>
      <td>Northcote</td>
      <td>155 Clarke St</td>
      <td>3</td>
      <td>h</td>
      <td>2750000.0</td>
      <td>VB</td>
      <td>Woodards</td>
      <td>14/10/2017</td>
      <td>5.3</td>
      <td>3070.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>634.0</td>
      <td>0.0</td>
      <td>1886.0</td>
      <td>Darebin City Council</td>
      <td>-37.77625</td>
      <td>144.99572</td>
      <td>Northern Metropolitan</td>
      <td>11364.0</td>
    </tr>
    <tr>
      <th>25086</th>
      <td>Glen Iris</td>
      <td>60 Hortense St</td>
      <td>4</td>
      <td>h</td>
      <td>2237500.0</td>
      <td>S</td>
      <td>Marshall</td>
      <td>21/10/2017</td>
      <td>7.3</td>
      <td>3146.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>650.0</td>
      <td>0.0</td>
      <td>1985.0</td>
      <td>Boroondara City Council</td>
      <td>-37.85776</td>
      <td>145.07998</td>
      <td>Southern Metropolitan</td>
      <td>10412.0</td>
    </tr>
    <tr>
      <th>25320</th>
      <td>Reservoir</td>
      <td>12 Kelverne St</td>
      <td>3</td>
      <td>h</td>
      <td>650000.0</td>
      <td>SP</td>
      <td>Barry</td>
      <td>21/10/2017</td>
      <td>12.0</td>
      <td>3073.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>491.0</td>
      <td>0.0</td>
      <td>1950.0</td>
      <td>Darebin City Council</td>
      <td>-37.71445</td>
      <td>144.98225</td>
      <td>Northern Metropolitan</td>
      <td>21650.0</td>
    </tr>
    <tr>
      <th>25352</th>
      <td>Roxburgh Park</td>
      <td>15 Donvale Av</td>
      <td>3</td>
      <td>h</td>
      <td>470000.0</td>
      <td>S</td>
      <td>Raine</td>
      <td>21/10/2017</td>
      <td>20.6</td>
      <td>3064.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>328.0</td>
      <td>0.0</td>
      <td>2004.0</td>
      <td>Hume City Council</td>
      <td>-37.61388</td>
      <td>144.92270</td>
      <td>Northern Metropolitan</td>
      <td>5833.0</td>
    </tr>
    <tr>
      <th>25376</th>
      <td>Spotswood</td>
      <td>104 Hudsons Rd</td>
      <td>2</td>
      <td>h</td>
      <td>1225000.0</td>
      <td>SP</td>
      <td>Greg</td>
      <td>21/10/2017</td>
      <td>6.2</td>
      <td>3015.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>361.0</td>
      <td>0.0</td>
      <td>1910.0</td>
      <td>Hobsons Bay City Council</td>
      <td>-37.82940</td>
      <td>144.88410</td>
      <td>Western Metropolitan</td>
      <td>1223.0</td>
    </tr>
    <tr>
      <th>25412</th>
      <td>Tarneit</td>
      <td>10 Discovery Dr</td>
      <td>4</td>
      <td>h</td>
      <td>585000.0</td>
      <td>S</td>
      <td>hockingstuart</td>
      <td>21/10/2017</td>
      <td>18.4</td>
      <td>3029.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>448.0</td>
      <td>0.0</td>
      <td>2010.0</td>
      <td>Wyndham City Council</td>
      <td>-37.84743</td>
      <td>144.71243</td>
      <td>Western Metropolitan</td>
      <td>10160.0</td>
    </tr>
    <tr>
      <th>25708</th>
      <td>Bundoora</td>
      <td>8 Oxford Dr</td>
      <td>3</td>
      <td>h</td>
      <td>770000.0</td>
      <td>S</td>
      <td>Ray</td>
      <td>28/10/2017</td>
      <td>12.1</td>
      <td>3083.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>551.0</td>
      <td>0.0</td>
      <td>1970.0</td>
      <td>Banyule City Council</td>
      <td>-37.69939</td>
      <td>145.06567</td>
      <td>Northern Metropolitan</td>
      <td>10175.0</td>
    </tr>
    <tr>
      <th>26343</th>
      <td>Preston</td>
      <td>148 Albert St</td>
      <td>3</td>
      <td>h</td>
      <td>833000.0</td>
      <td>S</td>
      <td>Stockdale</td>
      <td>28/10/2017</td>
      <td>8.4</td>
      <td>3072.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>501.0</td>
      <td>0.0</td>
      <td>1960.0</td>
      <td>Darebin City Council</td>
      <td>-37.73674</td>
      <td>145.02418</td>
      <td>Northern Metropolitan</td>
      <td>14577.0</td>
    </tr>
    <tr>
      <th>26633</th>
      <td>Craigieburn</td>
      <td>31 Yarcombe Cr</td>
      <td>4</td>
      <td>h</td>
      <td>540000.0</td>
      <td>S</td>
      <td>Barry</td>
      <td>4/11/2017</td>
      <td>20.6</td>
      <td>3064.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>541.0</td>
      <td>0.0</td>
      <td>1995.0</td>
      <td>Hume City Council</td>
      <td>-37.60800</td>
      <td>144.92530</td>
      <td>Northern Metropolitan</td>
      <td>15510.0</td>
    </tr>
    <tr>
      <th>27441</th>
      <td>Yallambie</td>
      <td>21 Lowan Av</td>
      <td>5</td>
      <td>h</td>
      <td>990000.0</td>
      <td>S</td>
      <td>Buckingham</td>
      <td>11/11/2017</td>
      <td>12.7</td>
      <td>3085.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>510.0</td>
      <td>0.0</td>
      <td>1985.0</td>
      <td>Banyule City Council</td>
      <td>-37.72040</td>
      <td>145.10880</td>
      <td>Northern Metropolitan</td>
      <td>1369.0</td>
    </tr>
    <tr>
      <th>27564</th>
      <td>Brighton East</td>
      <td>15 Bayview Rd</td>
      <td>3</td>
      <td>t</td>
      <td>950000.0</td>
      <td>VB</td>
      <td>Buxton</td>
      <td>18/11/2017</td>
      <td>10.3</td>
      <td>3187.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>217.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>Bayside City Council</td>
      <td>-37.90670</td>
      <td>145.02470</td>
      <td>Southern Metropolitan</td>
      <td>6938.0</td>
    </tr>
    <tr>
      <th>27587</th>
      <td>Brunswick West</td>
      <td>1/1 Duggan St</td>
      <td>2</td>
      <td>u</td>
      <td>420500.0</td>
      <td>SP</td>
      <td>Pagan</td>
      <td>18/11/2017</td>
      <td>5.2</td>
      <td>3055.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5497.0</td>
      <td>0.0</td>
      <td>2011.0</td>
      <td>Moreland City Council</td>
      <td>-37.75820</td>
      <td>144.94000</td>
      <td>Northern Metropolitan</td>
      <td>7082.0</td>
    </tr>
    <tr>
      <th>27629</th>
      <td>Carrum</td>
      <td>18 Church Rd</td>
      <td>4</td>
      <td>h</td>
      <td>980000.0</td>
      <td>S</td>
      <td>hockingstuart</td>
      <td>18/11/2017</td>
      <td>31.2</td>
      <td>3197.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>987.0</td>
      <td>0.0</td>
      <td>1960.0</td>
      <td>Kingston City Council</td>
      <td>-38.07920</td>
      <td>145.12760</td>
      <td>South-Eastern Metropolitan</td>
      <td>1989.0</td>
    </tr>
    <tr>
      <th>27922</th>
      <td>Lalor</td>
      <td>2 Orchid Ct</td>
      <td>5</td>
      <td>h</td>
      <td>591000.0</td>
      <td>S</td>
      <td>HAR</td>
      <td>18/11/2017</td>
      <td>16.3</td>
      <td>3075.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>636.0</td>
      <td>0.0</td>
      <td>1980.0</td>
      <td>Whittlesea City Council</td>
      <td>-37.67010</td>
      <td>145.00500</td>
      <td>Northern Metropolitan</td>
      <td>8279.0</td>
    </tr>
    <tr>
      <th>30914</th>
      <td>Wollert</td>
      <td>21 Dalwood Wy</td>
      <td>4</td>
      <td>h</td>
      <td>609000.0</td>
      <td>S</td>
      <td>HAR</td>
      <td>9/12/2017</td>
      <td>25.5</td>
      <td>3750.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>350.0</td>
      <td>0.0</td>
      <td>2015.0</td>
      <td>Whittlesea City Council</td>
      <td>-37.61031</td>
      <td>145.04010</td>
      <td>Northern Metropolitan</td>
      <td>2940.0</td>
    </tr>
    <tr>
      <th>31464</th>
      <td>Balwyn North</td>
      <td>4 Beverley Ct</td>
      <td>4</td>
      <td>t</td>
      <td>1190000.0</td>
      <td>VB</td>
      <td>Bekdon</td>
      <td>3/03/2018</td>
      <td>9.7</td>
      <td>3104.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>260.0</td>
      <td>0.0</td>
      <td>2017.0</td>
      <td>Boroondara City Council</td>
      <td>-37.79589</td>
      <td>145.09988</td>
      <td>Southern Metropolitan</td>
      <td>7809.0</td>
    </tr>
    <tr>
      <th>31509</th>
      <td>Blackburn South</td>
      <td>5 Abercromby Rd</td>
      <td>4</td>
      <td>h</td>
      <td>1400000.0</td>
      <td>VB</td>
      <td>Jellis</td>
      <td>3/03/2018</td>
      <td>13.4</td>
      <td>3130.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>560.0</td>
      <td>0.0</td>
      <td>1960.0</td>
      <td>Whitehorse City Council</td>
      <td>-37.83533</td>
      <td>145.14455</td>
      <td>Eastern Metropolitan</td>
      <td>4387.0</td>
    </tr>
    <tr>
      <th>31541</th>
      <td>Brunswick</td>
      <td>3 Austral Av</td>
      <td>4</td>
      <td>h</td>
      <td>1500000.0</td>
      <td>VB</td>
      <td>Nelson</td>
      <td>3/03/2018</td>
      <td>5.2</td>
      <td>3056.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>373.0</td>
      <td>0.0</td>
      <td>1940.0</td>
      <td>Moreland City Council</td>
      <td>-37.76289</td>
      <td>144.95552</td>
      <td>Northern Metropolitan</td>
      <td>11918.0</td>
    </tr>
    <tr>
      <th>31717</th>
      <td>Epping</td>
      <td>28 Bail St</td>
      <td>3</td>
      <td>h</td>
      <td>600000.0</td>
      <td>S</td>
      <td>hockingstuart</td>
      <td>3/03/2018</td>
      <td>19.6</td>
      <td>3076.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>461.0</td>
      <td>0.0</td>
      <td>2010.0</td>
      <td>Whittlesea City Council</td>
      <td>-37.62844</td>
      <td>145.00884</td>
      <td>Northern Metropolitan</td>
      <td>10926.0</td>
    </tr>
    <tr>
      <th>32403</th>
      <td>Roxburgh Park</td>
      <td>23 Wrigley Cr</td>
      <td>4</td>
      <td>h</td>
      <td>622000.0</td>
      <td>S</td>
      <td>Raine</td>
      <td>10/03/2018</td>
      <td>20.6</td>
      <td>3064.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>530.0</td>
      <td>0.0</td>
      <td>1998.0</td>
      <td>Hume City Council</td>
      <td>-37.62352</td>
      <td>144.93133</td>
      <td>Northern Metropolitan</td>
      <td>5833.0</td>
    </tr>
    <tr>
      <th>33397</th>
      <td>Greenvale</td>
      <td>26 Perugia Av</td>
      <td>4</td>
      <td>h</td>
      <td>677000.0</td>
      <td>S</td>
      <td>Ray</td>
      <td>17/03/2018</td>
      <td>20.4</td>
      <td>3059.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>312.0</td>
      <td>0.0</td>
      <td>2013.0</td>
      <td>Hume City Council</td>
      <td>-37.62439</td>
      <td>144.88629</td>
      <td>Northern Metropolitan</td>
      <td>4864.0</td>
    </tr>
    <tr>
      <th>33899</th>
      <td>Wollert</td>
      <td>40 Whitebark St</td>
      <td>4</td>
      <td>h</td>
      <td>615000.0</td>
      <td>S</td>
      <td>HAR</td>
      <td>17/03/2018</td>
      <td>25.5</td>
      <td>3750.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>392.0</td>
      <td>0.0</td>
      <td>2015.0</td>
      <td>Whittlesea City Council</td>
      <td>-37.61252</td>
      <td>145.04288</td>
      <td>Northern Metropolitan</td>
      <td>2940.0</td>
    </tr>
  </tbody>
</table>
<p>45 rows × 21 columns</p>
</div>




```python
housing[housing['Rooms']==0]
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
      <th>Suburb</th>
      <th>Address</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>SellerG</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>...</th>
      <th>Bathroom</th>
      <th>Car</th>
      <th>Landsize</th>
      <th>BuildingArea</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 21 columns</p>
</div>



BuildingArea is suspicious because of zero values. How could they be zero since they all have rooms? Some mistakes might be taken in the data entry. We choose to remove these observations.


```python
housing = housing[housing['BuildingArea']!=0]
```


```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8842 entries, 2 to 34856
    Data columns (total 21 columns):
    Suburb           8842 non-null object
    Address          8842 non-null object
    Rooms            8842 non-null int64
    Type             8842 non-null object
    Price            8842 non-null float64
    Method           8842 non-null object
    SellerG          8842 non-null object
    Date             8842 non-null object
    Distance         8842 non-null float64
    Postcode         8842 non-null float64
    Bedroom2         8842 non-null float64
    Bathroom         8842 non-null float64
    Car              8842 non-null float64
    Landsize         8842 non-null float64
    BuildingArea     8842 non-null float64
    YearBuilt        8842 non-null float64
    CouncilArea      8842 non-null object
    Lattitude        8842 non-null float64
    Longtitude       8842 non-null float64
    Regionname       8842 non-null object
    Propertycount    8842 non-null float64
    dtypes: float64(12), int64(1), object(8)
    memory usage: 1.5+ MB
    

**3. Data Reduction**
   
   Data reduction is necessary when there are too much redundant information in the dataset. In summary, common methods are:
   
   - Reducing the number of attributes
       - Data cube aggregation: applying roll-up, slice or dice operations.
       - Removing irrelevant attributes: attribute selection (filtering and wrapper methods), searching the attribute space (see Lecture 5: Attribute-oriented analysis).
       - Principle component analysis (numeric attributes only): searching for a lower dimensional space that can best represent the data
   - Reducing the number of attribute values
       - Binning (histograms): reducing the number of attributes by grouping them into intervals (bins).
       - Clustering: grouping values in clusters.
       - Aggregation or generalization
   - Reducing the number of tuples
       - Sampling
       
   Reference: http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html
   

In this case, Address and sellerG (real estate agent) seems to be unnecessary currently. In the future, natural language processing might help include in the model. Also, bedroom2 is scraped from different sources. We chose to ignore this column.  


```python
housing = housing.drop(['Address','SellerG'],axis = 1)    #axis=1 indicate the column name
```

### 4. Feature Engineering

Feature engineering is important to dig more inforamtion from row data. We just translate some attributes into new ones that make more sense in the reality. For example, in the case, we would translate YearBuilt into **Age of property**. Date sold can also be split into three attributes: **Year, Month and Weekday**. 

Let's do it!


```python
housing['Age'] = 2018-housing['YearBuilt']
```


```python
housing[housing['Age']<0]
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
      <th>Suburb</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>...</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
      <th>Age</th>
      <th>Year</th>
      <th>Month</th>
      <th>Weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33033</th>
      <td>Bentleigh</td>
      <td>3</td>
      <td>h</td>
      <td>1100000.0</td>
      <td>VB</td>
      <td>2018-03-17</td>
      <td>11.4</td>
      <td>3204.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2019.0</td>
      <td>Glen Eira City Council</td>
      <td>-37.92963</td>
      <td>145.03666</td>
      <td>Southern Metropolitan</td>
      <td>6795.0</td>
      <td>-1.0</td>
      <td>2018</td>
      <td>3</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 23 columns</p>
</div>



** Notice Here: There is one property to be built in 2019 **

We are using groupby function to see the average price VS age of property


```python
plt.figure(figsize=(10,6))
housing.groupby('Age')['Price'].mean().plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17065f3c128>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_57_1.png)


Here, we found a age of 800. It could be true, but it seems an outlier. We choose to take it out.


```python
housing[housing['Age']>400]
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
      <th>Suburb</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>...</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
      <th>Age</th>
      <th>Year</th>
      <th>Month</th>
      <th>Weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16424</th>
      <td>Mount Waverley</td>
      <td>3</td>
      <td>h</td>
      <td>1200000.0</td>
      <td>VB</td>
      <td>2017-06-24</td>
      <td>14.2</td>
      <td>3149.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1196.0</td>
      <td>Monash City Council</td>
      <td>-37.86788</td>
      <td>145.12121</td>
      <td>Eastern Metropolitan</td>
      <td>13366.0</td>
      <td>822.0</td>
      <td>2017</td>
      <td>6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 23 columns</p>
</div>




```python
housing = housing[housing['Age']<500]
```

Then we visualize again. It should be good now.


```python
plt.figure(figsize=(10,6))
ax=housing.groupby('Age')['Price'].mean().plot()
ax.set_ylabel('Price')
```




    Text(0,0.5,'Price')




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_62_1.png)


**Interesting Finding!**

Then we transform Date column!


```python
housing['Date']=pd.to_datetime(housing['Date'],infer_datetime_format=True)  ## transform Date column to datetime object
```


```python
housing['Year']=pd.DatetimeIndex(housing['Date']).year
housing['Month']=pd.DatetimeIndex(housing['Date']).month
housing['Weekday']=pd.DatetimeIndex(housing['Date']).weekday
```


```python
housing.head()
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
      <th>Suburb</th>
      <th>Rooms</th>
      <th>Type</th>
      <th>Price</th>
      <th>Method</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Postcode</th>
      <th>Bedroom2</th>
      <th>Bathroom</th>
      <th>...</th>
      <th>YearBuilt</th>
      <th>CouncilArea</th>
      <th>Lattitude</th>
      <th>Longtitude</th>
      <th>Regionname</th>
      <th>Propertycount</th>
      <th>Age</th>
      <th>Year</th>
      <th>Month</th>
      <th>Weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>1035000.0</td>
      <td>S</td>
      <td>2016-04-02</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1900.0</td>
      <td>Yarra City Council</td>
      <td>-37.8079</td>
      <td>144.9934</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>118.0</td>
      <td>2016</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abbotsford</td>
      <td>3</td>
      <td>h</td>
      <td>1465000.0</td>
      <td>SP</td>
      <td>2017-04-03</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1900.0</td>
      <td>Yarra City Council</td>
      <td>-37.8093</td>
      <td>144.9944</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>118.0</td>
      <td>2017</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Abbotsford</td>
      <td>4</td>
      <td>h</td>
      <td>1600000.0</td>
      <td>VB</td>
      <td>2016-04-06</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2014.0</td>
      <td>Yarra City Council</td>
      <td>-37.8072</td>
      <td>144.9941</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>4.0</td>
      <td>2016</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Abbotsford</td>
      <td>3</td>
      <td>h</td>
      <td>1876000.0</td>
      <td>S</td>
      <td>2016-07-05</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1910.0</td>
      <td>Yarra City Council</td>
      <td>-37.8024</td>
      <td>144.9993</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>108.0</td>
      <td>2016</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Abbotsford</td>
      <td>2</td>
      <td>h</td>
      <td>1636000.0</td>
      <td>S</td>
      <td>2016-08-10</td>
      <td>2.5</td>
      <td>3067.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1890.0</td>
      <td>Yarra City Council</td>
      <td>-37.8060</td>
      <td>144.9954</td>
      <td>Northern Metropolitan</td>
      <td>4019.0</td>
      <td>128.0</td>
      <td>2016</td>
      <td>8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
sns.factorplot(y='Price',x='Month',kind='bar',data=housing)
```




    <seaborn.axisgrid.FacetGrid at 0x170620c41d0>




![png](output_67_1.png)



```python
sns.factorplot(y='Price',x='Weekday',kind='bar',data=housing) 

```




    <seaborn.axisgrid.FacetGrid at 0x17066119d30>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_68_1.png)



```python
sns.factorplot(y='Price',x='Year',kind='bar',data=housing)
```




    <seaborn.axisgrid.FacetGrid at 0x170627f4358>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/housing/output_69_1.png)


## Conclusion

1. Data preprocessing is very important.
2. More advanced method in outliers detection and missing values handling is to be studied in the future
   - I am really not sure what outliers should be taken into account. Price has many extreme values from boxplot. Should we take all of them off?
   - Missing values had better be taken care of carefully. Here, I just drop them.
3. Visualization will be introduced next time.

Thank you!
