---
layout: post
title: Customer Segmentation using K-means clustering 
subtitle: Clustering for bikeshops
gh-repo: zigzagjie/Data-Science
gh-badge: [star, fork, follow]
tags: [Pandas,Clustering]
---


This is my solution for homework 4 of 95-851 Making Products Count: Data Science for Product Management. This is a good practice for data manipulation and KMeans clustering in sklearn.

The goal of this project is to group bike shops with similar bikes sales pattern. Based on the assumption that bikes are selected by its model, price, and other features, customers may have different preferences towards bike shops. We'll find this pattern using K means Clustering.

We are given three datasets: bikes, order, bikeshops. The first step is to merge them and aggregate bikes order for all 30 bike shops. Then, to scale the quantity, we transform the quantity to the relative quantity(percentage). Last, we choose the right number of clusters to perform K-means Clustering.




```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

```

## Step 1: Read Datasets


```python
bike = pd.ExcelFile("bikes.xlsx").parse("Sheet1")
shop = pd.ExcelFile("bikeshops.xlsx").parse("Sheet1")
order = pd.ExcelFile("orders.xlsx").parse("Sheet1")
```


```python
order.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15644 entries, 1 to 15644
    Data columns (total 6 columns):
    order.id       15644 non-null int64
    order.line     15644 non-null int64
    order.date     15644 non-null datetime64[ns]
    customer.id    15644 non-null int64
    product.id     15644 non-null int64
    quantity       15644 non-null int64
    dtypes: datetime64[ns](1), int64(5)
    memory usage: 855.5 KB
    


```python
bike.head()
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
      <th>bike.id</th>
      <th>model</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Supersix Evo Black Inc.</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>12790</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Supersix Evo Hi-Mod Team</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>10660</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Supersix Evo Hi-Mod Dura Ace 1</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>7990</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Supersix Evo Hi-Mod Dura Ace 2</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>5330</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Supersix Evo Hi-Mod Utegra</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>4260</td>
    </tr>
  </tbody>
</table>
</div>




```python
shop.head()
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
      <th>bikeshop.id</th>
      <th>bikeshop.name</th>
      <th>bikeshop.city</th>
      <th>bikeshop.state</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Pittsburgh Mountain Machines</td>
      <td>Pittsburgh</td>
      <td>PA</td>
      <td>40.440625</td>
      <td>-79.995886</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ithaca Mountain Climbers</td>
      <td>Ithaca</td>
      <td>NY</td>
      <td>42.443961</td>
      <td>-76.501881</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Columbus Race Equipment</td>
      <td>Columbus</td>
      <td>OH</td>
      <td>39.961176</td>
      <td>-82.998794</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Detroit Cycles</td>
      <td>Detroit</td>
      <td>MI</td>
      <td>42.331427</td>
      <td>-83.045754</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Cincinnati Speed</td>
      <td>Cincinnati</td>
      <td>OH</td>
      <td>39.103118</td>
      <td>-84.512020</td>
    </tr>
  </tbody>
</table>
</div>




```python
order.head()
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
      <th>order.id</th>
      <th>order.line</th>
      <th>order.date</th>
      <th>customer.id</th>
      <th>product.id</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2011-01-07</td>
      <td>2</td>
      <td>48</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>2011-01-07</td>
      <td>2</td>
      <td>52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>2011-01-10</td>
      <td>10</td>
      <td>76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>2011-01-10</td>
      <td>10</td>
      <td>52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>1</td>
      <td>2011-01-10</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Step 2: Merge three datasets


```python
order_shop=pd.merge(left=order, right=shop, how='outer', left_on="customer.id", right_on="bikeshop.id")
order_final = pd.merge(left=order_shop,right = bike,left_on = "product.id",right_on="bike.id")
```


```python
order_final.head()
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
      <th>order.id</th>
      <th>order.line</th>
      <th>order.date</th>
      <th>customer.id</th>
      <th>product.id</th>
      <th>quantity</th>
      <th>bikeshop.id</th>
      <th>bikeshop.name</th>
      <th>bikeshop.city</th>
      <th>bikeshop.state</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>bike.id</th>
      <th>model</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2011-01-07</td>
      <td>2</td>
      <td>48</td>
      <td>1</td>
      <td>2</td>
      <td>Ithaca Mountain Climbers</td>
      <td>Ithaca</td>
      <td>NY</td>
      <td>42.443961</td>
      <td>-76.501881</td>
      <td>48</td>
      <td>Jekyll Carbon 2</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>6070</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132</td>
      <td>6</td>
      <td>2011-05-13</td>
      <td>2</td>
      <td>48</td>
      <td>1</td>
      <td>2</td>
      <td>Ithaca Mountain Climbers</td>
      <td>Ithaca</td>
      <td>NY</td>
      <td>42.443961</td>
      <td>-76.501881</td>
      <td>48</td>
      <td>Jekyll Carbon 2</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>6070</td>
    </tr>
    <tr>
      <th>2</th>
      <td>507</td>
      <td>2</td>
      <td>2012-06-26</td>
      <td>2</td>
      <td>48</td>
      <td>1</td>
      <td>2</td>
      <td>Ithaca Mountain Climbers</td>
      <td>Ithaca</td>
      <td>NY</td>
      <td>42.443961</td>
      <td>-76.501881</td>
      <td>48</td>
      <td>Jekyll Carbon 2</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>6070</td>
    </tr>
    <tr>
      <th>3</th>
      <td>528</td>
      <td>18</td>
      <td>2012-07-16</td>
      <td>2</td>
      <td>48</td>
      <td>1</td>
      <td>2</td>
      <td>Ithaca Mountain Climbers</td>
      <td>Ithaca</td>
      <td>NY</td>
      <td>42.443961</td>
      <td>-76.501881</td>
      <td>48</td>
      <td>Jekyll Carbon 2</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>6070</td>
    </tr>
    <tr>
      <th>4</th>
      <td>691</td>
      <td>13</td>
      <td>2013-02-05</td>
      <td>2</td>
      <td>48</td>
      <td>1</td>
      <td>2</td>
      <td>Ithaca Mountain Climbers</td>
      <td>Ithaca</td>
      <td>NY</td>
      <td>42.443961</td>
      <td>-76.501881</td>
      <td>48</td>
      <td>Jekyll Carbon 2</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>6070</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Convert Unit Price


```python
#order_final.groupby(['model','category1','category2','frame'])['price'].median().median()
#order_final.price.median()
#order_final.groupby(['category1','category2','frame'])['price'].unique().apply(np.median)
order_final.groupby('model').price.unique().apply(lambda x:x[0]).median()
#np.sort(order_final.price.unique())
#np.sort(order_final.price.unique())
```




    3200.0



#### Median Price for all the models are 3200. So we categorize price lower than and eqaul to 3200 as low, otherwise high. 


```python
#order_final["category median"]=order_final.groupby(['category1','category2','frame'])['price'].transform(lambda x:np.median(x.unique()))
#order_final['unitPrice']=(order_final['price']>order_final["category median"]).apply(lambda x: "high" if x==True else "low")
order_final['unitPrice']=order_final['price'].apply(lambda x: 'high' if x>3200 else 'low')
```

## Step 4: Obtain pivot table


```python
summary=order_final.groupby(['bikeshop.name', 'model', 'category1', 'category2', 'frame', 'price','unitPrice']).quantity.sum().reset_index()
customers = summary[["bikeshop.name","model","quantity"]].pivot(index="bikeshop.name",columns = "model",values="quantity").fillna(0).reset_index().rename_axis(None,1)
```


```python
summary.head()
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
      <th>bikeshop.name</th>
      <th>model</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>price</th>
      <th>unitPrice</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albuquerque Cycles</td>
      <td>Bad Habit 1</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>3200</td>
      <td>low</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albuquerque Cycles</td>
      <td>Bad Habit 2</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>2660</td>
      <td>low</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albuquerque Cycles</td>
      <td>Beast of the East 1</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>2770</td>
      <td>low</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albuquerque Cycles</td>
      <td>Beast of the East 2</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>2130</td>
      <td>low</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albuquerque Cycles</td>
      <td>Beast of the East 3</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>1620</td>
      <td>low</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Get ratio

scaling


```python
customers=customers.set_index("bikeshop.name")
customers['sum']=customers.sum(axis=1)
customers=customers.div(customers["sum"], axis=0).drop('sum',axis=1)
```


```python
customers.head()
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
      <th>Bad Habit 1</th>
      <th>Bad Habit 2</th>
      <th>Beast of the East 1</th>
      <th>Beast of the East 2</th>
      <th>Beast of the East 3</th>
      <th>CAAD Disc Ultegra</th>
      <th>CAAD12 105</th>
      <th>CAAD12 Black Inc</th>
      <th>CAAD12 Disc 105</th>
      <th>CAAD12 Disc Dura Ace</th>
      <th>...</th>
      <th>Synapse Sora</th>
      <th>Trail 1</th>
      <th>Trail 2</th>
      <th>Trail 3</th>
      <th>Trail 4</th>
      <th>Trail 5</th>
      <th>Trigger Carbon 1</th>
      <th>Trigger Carbon 2</th>
      <th>Trigger Carbon 3</th>
      <th>Trigger Carbon 4</th>
    </tr>
    <tr>
      <th>bikeshop.name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albuquerque Cycles</th>
      <td>0.017483</td>
      <td>0.006993</td>
      <td>0.01049</td>
      <td>0.010490</td>
      <td>0.003497</td>
      <td>0.013986</td>
      <td>0.006993</td>
      <td>0.000000</td>
      <td>0.013986</td>
      <td>0.048951</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.003497</td>
      <td>0.006993</td>
      <td>0.017483</td>
      <td>0.010490</td>
      <td>0.006993</td>
      <td>0.003497</td>
      <td>0.006993</td>
      <td>0.006993</td>
    </tr>
    <tr>
      <th>Ann Arbor Speed</th>
      <td>0.006645</td>
      <td>0.009967</td>
      <td>0.01495</td>
      <td>0.009967</td>
      <td>0.003322</td>
      <td>0.026578</td>
      <td>0.014950</td>
      <td>0.016611</td>
      <td>0.014950</td>
      <td>0.008306</td>
      <td>...</td>
      <td>0.009967</td>
      <td>0.009967</td>
      <td>0.014950</td>
      <td>0.009967</td>
      <td>0.003322</td>
      <td>0.011628</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011628</td>
    </tr>
    <tr>
      <th>Austin Cruisers</th>
      <td>0.008130</td>
      <td>0.004065</td>
      <td>0.00813</td>
      <td>0.008130</td>
      <td>0.000000</td>
      <td>0.020325</td>
      <td>0.020325</td>
      <td>0.004065</td>
      <td>0.024390</td>
      <td>0.008130</td>
      <td>...</td>
      <td>0.020325</td>
      <td>0.016260</td>
      <td>0.016260</td>
      <td>0.016260</td>
      <td>0.008130</td>
      <td>0.008130</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016260</td>
    </tr>
    <tr>
      <th>Cincinnati Speed</th>
      <td>0.005115</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015345</td>
      <td>0.010230</td>
      <td>0.015345</td>
      <td>0.007673</td>
      <td>0.017903</td>
      <td>...</td>
      <td>0.012788</td>
      <td>0.000000</td>
      <td>0.002558</td>
      <td>0.002558</td>
      <td>0.002558</td>
      <td>0.000000</td>
      <td>0.010230</td>
      <td>0.007673</td>
      <td>0.010230</td>
      <td>0.020460</td>
    </tr>
    <tr>
      <th>Columbus Race Equipment</th>
      <td>0.010152</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.005076</td>
      <td>0.002538</td>
      <td>0.010152</td>
      <td>0.027919</td>
      <td>0.027919</td>
      <td>0.025381</td>
      <td>0.012690</td>
      <td>...</td>
      <td>0.015228</td>
      <td>0.002538</td>
      <td>0.002538</td>
      <td>0.005076</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010152</td>
      <td>0.005076</td>
      <td>0.017766</td>
      <td>0.005076</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 97 columns</p>
</div>



## Step 5: KMeans Clustering

I used elbow method and silhouette analysis to determine the number of clusters. It is hard to tell from elbow method but silhouette analysis clearly suggests K=5.


```python
from sklearn.cluster import KMeans
inertias = []
for i in range(4,9):
    kmean = KMeans(n_clusters=i,random_state=20,n_init=50)
    kmean.fit(customers)
    inertias.append(kmean.inertia_)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(range(4,9),inertias)
    
```




    [<matplotlib.lines.Line2D at 0x148f94dc0b8>]




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/bike/output1.png)



```python
from sklearn.metrics import silhouette_samples, silhouette_score
silhouette = []
for i in range(4,9):
    clusterer = KMeans(n_clusters=i, random_state=20,n_init=50)
    cluster_labels = clusterer.fit_predict(customers)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(customers, cluster_labels)
    silhouette.append(silhouette_avg)
    #print("For n_clusters =",i,
    #      "The average silhouette_score is :", silhouette_avg)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(range(4,9),silhouette)
```




    [<matplotlib.lines.Line2D at 0x148f9c42080>]




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/bike/output2.png)


# Step 6: Cluster Analysis


```python
clusterer = KMeans(n_clusters=5, random_state=20,n_init=50)
cluster_labels = clusterer.fit_predict(customers)
```


```python
customers['group'] = cluster_labels
```


```python
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(customers.groupby("group").mean().T)
#plt.figure(figsize=(10,10))
#sns.heatmap(customers.groupby("group").mean())
#customers.groupby("group").mean()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x148f5f982e8>




![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/bike/output3.png)



```python
pd.set_option('max_colwidth', 800)
pd.DataFrame(customers.reset_index().groupby("group")["bikeshop.name"].unique())
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
      <th>bikeshop.name</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Philadelphia Bike Shop, San Antonio Bike Shop]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Ann Arbor Speed, Austin Cruisers, Indianapolis Velocipedes, Miami Race Equipment, Nashville Cruisers, New Orleans Velocipedes, Oklahoma City Race Equipment, Seattle Race Equipment]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Ithaca Mountain Climbers, Pittsburgh Mountain Machines, Tampa 29ers]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Cincinnati Speed, Columbus Race Equipment, Las Vegas Cycles, Louisville Race Equipment, San Francisco Cruisers, Wichita Speed]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Albuquerque Cycles, Dallas Cycles, Denver Bike Shop, Detroit Cycles, Kansas City 29ers, Los Angeles Cycles, Minneapolis Bike Shop, New York Cycles, Phoenix Bi-peds, Portland Bi-peds, Providence Bi-peds]</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratio = lambda x: x.div(x.sum(axis=1),axis=0)
```


```python
groups = customers.reset_index()[['bikeshop.name','group']]
final=pd.merge(left = summary,right=groups,on="bikeshop.name")
```


```python
bike_info = summary[['model','category1','category2','frame','unitPrice']].drop_duplicates()
```

## Segmentation Analysis

Now that we get four segmented groups, we need find out the distinctions between them. It is a good way to validate our clustering results. 

Since one shop has sold lots of different models of bikes, we just study on the top 10 bikes sold in each cluster. 


```python
def top10(groupN):
    return pd.DataFrame(customers[customers.group==groupN].mean().sort_values(ascending=False).iloc[1:]).reset_index().rename(columns={'index': 'model',0:'ratio'}).head(10).merge(bike_info,on='model')
top10(groupN=0)
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
      <th>model</th>
      <th>ratio</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>unitPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trigger Carbon 4</td>
      <td>0.030415</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CAAD12 105</td>
      <td>0.027079</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beast of the East 3</td>
      <td>0.026333</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trail 1</td>
      <td>0.024940</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bad Habit 1</td>
      <td>0.022998</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>5</th>
      <td>F-Si Carbon 4</td>
      <td>0.022449</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CAAD12 Disc 105</td>
      <td>0.020957</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Synapse Disc 105</td>
      <td>0.020957</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Trail 2</td>
      <td>0.020309</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Synapse Carbon Disc 105</td>
      <td>0.019662</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 0

Features of bikes: relatively low price, mixed category1 and frames


```python
top10(groupN=1)
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
      <th>model</th>
      <th>ratio</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>unitPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Synapse Disc Tiagra</td>
      <td>0.024608</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CAAD12 Red</td>
      <td>0.022889</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Synapse Sora</td>
      <td>0.022305</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Slice Ultegra</td>
      <td>0.022077</td>
      <td>Road</td>
      <td>Triathalon</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Supersix Evo Ultegra 3</td>
      <td>0.021857</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Synapse Disc 105</td>
      <td>0.021122</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Synapse Carbon Ultegra 4</td>
      <td>0.020372</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CAAD12 Ultegra</td>
      <td>0.020073</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Slice Ultegra D12</td>
      <td>0.019531</td>
      <td>Road</td>
      <td>Triathalon</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Supersix Evo Ultegra 4</td>
      <td>0.019435</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 1

Features of bikes: relatively low price, Road bikes and mixed-type frames


```python
top10(groupN=2)
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
      <th>model</th>
      <th>ratio</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>unitPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Scalpel-Si Carbon 3</td>
      <td>0.034269</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jekyll Carbon 4</td>
      <td>0.030282</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Scalpel 29 Carbon Race</td>
      <td>0.028039</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trigger Carbon 3</td>
      <td>0.025935</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Habit Carbon 2</td>
      <td>0.023375</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Trigger Carbon 4</td>
      <td>0.023261</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Catalyst 4</td>
      <td>0.021564</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jekyll Carbon 2</td>
      <td>0.021079</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Supersix Evo Hi-Mod Dura Ace 2</td>
      <td>0.021058</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Trigger Carbon 2</td>
      <td>0.021043</td>
      <td>Mountain</td>
      <td>Over Mountain</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 2

Features of bikes: mixed level price, mountain bikes with Carbon frames


```python
top10(groupN=3)
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
      <th>model</th>
      <th>ratio</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>unitPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Synapse Hi-Mod Disc Red</td>
      <td>0.024675</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Slice Hi-Mod Black Inc.</td>
      <td>0.023494</td>
      <td>Road</td>
      <td>Triathalon</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Supersix Evo Hi-Mod Dura Ace 1</td>
      <td>0.023053</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Slice Hi-Mod Dura Ace D12</td>
      <td>0.022961</td>
      <td>Road</td>
      <td>Triathalon</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Synapse Hi-Mod Dura Ace</td>
      <td>0.021672</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CAAD12 Red</td>
      <td>0.021196</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Synapse Carbon Disc Ultegra</td>
      <td>0.020239</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Supersix Evo Ultegra 3</td>
      <td>0.020187</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Supersix Evo Hi-Mod Utegra</td>
      <td>0.019819</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Synapse Hi-Mod Disc Black Inc.</td>
      <td>0.019755</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>high</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 3

Features of bikes: relatively high price, Road bikes and carbon frames


```python
top10(groupN=4)
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
      <th>model</th>
      <th>ratio</th>
      <th>category1</th>
      <th>category2</th>
      <th>frame</th>
      <th>unitPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F-Si 2</td>
      <td>0.021705</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Catalyst 3</td>
      <td>0.019743</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F-Si Carbon 4</td>
      <td>0.017716</td>
      <td>Mountain</td>
      <td>Cross Country Race</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trail 5</td>
      <td>0.016265</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CAAD8 Sora</td>
      <td>0.016141</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CAAD12 Disc 105</td>
      <td>0.015987</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CAAD8 105</td>
      <td>0.015796</td>
      <td>Road</td>
      <td>Elite Road</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Habit 4</td>
      <td>0.015531</td>
      <td>Mountain</td>
      <td>Trail</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Synapse Carbon Disc 105</td>
      <td>0.015407</td>
      <td>Road</td>
      <td>Endurance Road</td>
      <td>Carbon</td>
      <td>low</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Trail 3</td>
      <td>0.014985</td>
      <td>Mountain</td>
      <td>Sport</td>
      <td>Aluminum</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 4

Features of bikes: relatively low price, mixed-type bikes and Aluminum frames

# Trying other methods of clustering

Using hierachical clustering


```python
customers = customers.drop('group',axis=1)
```


```python
customers.head()
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
      <th>Bad Habit 1</th>
      <th>Bad Habit 2</th>
      <th>Beast of the East 1</th>
      <th>Beast of the East 2</th>
      <th>Beast of the East 3</th>
      <th>CAAD Disc Ultegra</th>
      <th>CAAD12 105</th>
      <th>CAAD12 Black Inc</th>
      <th>CAAD12 Disc 105</th>
      <th>CAAD12 Disc Dura Ace</th>
      <th>...</th>
      <th>Synapse Sora</th>
      <th>Trail 1</th>
      <th>Trail 2</th>
      <th>Trail 3</th>
      <th>Trail 4</th>
      <th>Trail 5</th>
      <th>Trigger Carbon 1</th>
      <th>Trigger Carbon 2</th>
      <th>Trigger Carbon 3</th>
      <th>Trigger Carbon 4</th>
    </tr>
    <tr>
      <th>bikeshop.name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albuquerque Cycles</th>
      <td>0.017483</td>
      <td>0.006993</td>
      <td>0.01049</td>
      <td>0.010490</td>
      <td>0.003497</td>
      <td>0.013986</td>
      <td>0.006993</td>
      <td>0.000000</td>
      <td>0.013986</td>
      <td>0.048951</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.003497</td>
      <td>0.006993</td>
      <td>0.017483</td>
      <td>0.010490</td>
      <td>0.006993</td>
      <td>0.003497</td>
      <td>0.006993</td>
      <td>0.006993</td>
    </tr>
    <tr>
      <th>Ann Arbor Speed</th>
      <td>0.006645</td>
      <td>0.009967</td>
      <td>0.01495</td>
      <td>0.009967</td>
      <td>0.003322</td>
      <td>0.026578</td>
      <td>0.014950</td>
      <td>0.016611</td>
      <td>0.014950</td>
      <td>0.008306</td>
      <td>...</td>
      <td>0.009967</td>
      <td>0.009967</td>
      <td>0.014950</td>
      <td>0.009967</td>
      <td>0.003322</td>
      <td>0.011628</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011628</td>
    </tr>
    <tr>
      <th>Austin Cruisers</th>
      <td>0.008130</td>
      <td>0.004065</td>
      <td>0.00813</td>
      <td>0.008130</td>
      <td>0.000000</td>
      <td>0.020325</td>
      <td>0.020325</td>
      <td>0.004065</td>
      <td>0.024390</td>
      <td>0.008130</td>
      <td>...</td>
      <td>0.020325</td>
      <td>0.016260</td>
      <td>0.016260</td>
      <td>0.016260</td>
      <td>0.008130</td>
      <td>0.008130</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016260</td>
    </tr>
    <tr>
      <th>Cincinnati Speed</th>
      <td>0.005115</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015345</td>
      <td>0.010230</td>
      <td>0.015345</td>
      <td>0.007673</td>
      <td>0.017903</td>
      <td>...</td>
      <td>0.012788</td>
      <td>0.000000</td>
      <td>0.002558</td>
      <td>0.002558</td>
      <td>0.002558</td>
      <td>0.000000</td>
      <td>0.010230</td>
      <td>0.007673</td>
      <td>0.010230</td>
      <td>0.020460</td>
    </tr>
    <tr>
      <th>Columbus Race Equipment</th>
      <td>0.010152</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.005076</td>
      <td>0.002538</td>
      <td>0.010152</td>
      <td>0.027919</td>
      <td>0.027919</td>
      <td>0.025381</td>
      <td>0.012690</td>
      <td>...</td>
      <td>0.015228</td>
      <td>0.002538</td>
      <td>0.002538</td>
      <td>0.005076</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010152</td>
      <td>0.005076</td>
      <td>0.017766</td>
      <td>0.005076</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 97 columns</p>
</div>




```python
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

linked = linkage(customers, 'ward')

#labelList = customers.index
labelList = range(0,30)

plt.figure(figsize=(20, 10))  
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)

plt.axhline(y=0.16, color='r', linestyle='-')

plt.show()  
```


![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/bike/output4.png)


#### 4 clusters are chosen here.


```python
newclusters=[]
group1=[2,8,27,16,1,17,14,19]
group2 = [20,25,0,5,12,6,10,18,7,24,15,21,23]
group3 = [3,13,11,29,4,26]
group4 = [28,9,22]

newclusters.append([['group0'],customers.index[group1].tolist()])
newclusters.append([['group1'],customers.index[group2].tolist()])
newclusters.append([['group2'],customers.index[group3].tolist()])
newclusters.append([['group3'],customers.index[group4].tolist()])

```


```python
pd.DataFrame(np.array(newclusters))
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[group0]</td>
      <td>[Austin Cruisers, Indianapolis Velocipedes, Seattle Race Equipment, Nashville Cruisers, Ann Arbor Speed, New Orleans Velocipedes, Miami Race Equipment, Oklahoma City Race Equipment]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[group1]</td>
      <td>[Philadelphia Bike Shop, San Antonio Bike Shop, Albuquerque Cycles, Dallas Cycles, Los Angeles Cycles, Denver Bike Shop, Kansas City 29ers, New York Cycles, Detroit Cycles, Providence Bi-peds, Minneapolis Bike Shop, Phoenix Bi-peds, Portland Bi-peds]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[group2]</td>
      <td>[Cincinnati Speed, Louisville Race Equipment, Las Vegas Cycles, Wichita Speed, Columbus Race Equipment, San Francisco Cruisers]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[group3]</td>
      <td>[Tampa 29ers, Ithaca Mountain Climbers, Pittsburgh Mountain Machines]</td>
    </tr>
  </tbody>
</table>
</div>



above is new clustering grouping

below is previous clsutering grouping
### Compared to the results from KMeans, it was found cluster 0 and 4 were combined. 
above group0 --> below group 1

above group1 --> below group0+below group4

above group2 --> below group3

above group3 --> below group2


```python
pd.DataFrame(customers.reset_index().groupby("group")["bikeshop.name"].unique())
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
      <th>bikeshop.name</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Philadelphia Bike Shop, San Antonio Bike Shop]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Ann Arbor Speed, Austin Cruisers, Indianapolis Velocipedes, Miami Race Equipment, Nashville Cruisers, New Orleans Velocipedes, Oklahoma City Race Equipment, Seattle Race Equipment]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Ithaca Mountain Climbers, Pittsburgh Mountain Machines, Tampa 29ers]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Cincinnati Speed, Columbus Race Equipment, Las Vegas Cycles, Louisville Race Equipment, San Francisco Cruisers, Wichita Speed]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Albuquerque Cycles, Dallas Cycles, Denver Bike Shop, Detroit Cycles, Kansas City 29ers, Los Angeles Cycles, Minneapolis Bike Shop, New York Cycles, Phoenix Bi-peds, Portland Bi-peds, Providence Bi-peds]</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 0 and 4
Cluster 0 --> Features of bikes: relatively low price, mixed-type bikes and frames
Cluster 4 --> Features of bikes: relatively low price, mixed-type bikes and Aluminum frames

Cluster 0 and 4 are very similary and they can be combined into big groups: relatively low price, mixed-type bikes and mixed-type frames.

In hierachical clustering, with different criteria to cut tree, the results can be different. Thus, in this sense, cluster 0 and cluster 4 can be combined because of high similarity/closeness.

###  ------------------------------------------------   The End   ----------------------------------------------------------------------
