---
layout: post
title: Customer Longtime Value in Python
subtitle: Python version to calculate customer longtime value
gh-repo: zigzagjie/Data-Science
gh-badge: [star, fork, follow]
tags: [Pandas,PM]
---

# Customer Longtime Value in Python

This notebook goes through the calculation in [Practical Guide to Calculating Customer Lifetime Value (CLV)](https://gormanalysis.com/practical-guide-to-calculating-customer-lifetime-value-clv/) with Python. It mainly applies pandas dataframe to deal with the dataset.

**Background: Customer Lifetime Value**

* It is a prediction of the net profit attributed to the entire future relationship with a customer
* Also defined as the dollar value of a customer relationship based on the projected future cash flows from the customer relationship
* Represents an upper limit on spending to acquire new customers

## Step 1. Load the transaction dataset




```python
 customer.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4186 entries, 0 to 4185
    Data columns (total 4 columns):
    TransactionID      4186 non-null int64
    TransactionDate    4186 non-null datetime64[ns]
    CustomerID         4186 non-null int64
    Amount             4186 non-null float64
    dtypes: datetime64[ns](1), float64(1), int64(2)
    memory usage: 130.9 KB
    


```python
 customer.head()
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
      <th>TransactionID</th>
      <th>TransactionDate</th>
      <th>CustomerID</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2012-09-04</td>
      <td>1</td>
      <td>20.26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2012-05-15</td>
      <td>2</td>
      <td>10.87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2014-05-23</td>
      <td>2</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2014-10-24</td>
      <td>2</td>
      <td>10.48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2012-10-13</td>
      <td>2</td>
      <td>3.94</td>
    </tr>
  </tbody>
</table>
</div>



There are some outliers in the dataset. We need to remove them. The details are not important for this analysis.


```python
 #customer_new is the new dataset without outliers
 customer_new = customer[(customer.Amount<1000)&(customer.Amount>0)]
```

## Step 2. Determine origin year of customers


```python
 customer_new.head()
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
      <th>TransactionID</th>
      <th>TransactionDate</th>
      <th>CustomerID</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2012-09-04</td>
      <td>1</td>
      <td>20.26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2012-05-15</td>
      <td>2</td>
      <td>10.87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2014-05-23</td>
      <td>2</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2014-10-24</td>
      <td>2</td>
      <td>10.48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2012-10-13</td>
      <td>2</td>
      <td>3.94</td>
    </tr>
  </tbody>
</table>
</div>



**We need find each group the customer assigned in. The rule is to record the earliest transaction with each customer**


```python
# Get the earliest year each customer have made transactions, stored in group dataframe
group = customer_new.groupby('CustomerID').TransactionDate.unique().apply(min).reset_index()
```


```python
group.head()
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
      <th>CustomerID</th>
      <th>TransactionDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2012-09-04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2012-05-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2012-11-26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2015-07-07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2015-01-24</td>
    </tr>
  </tbody>
</table>
</div>




```python
# extract years from the dataframe
group['Group'] = group.TransactionDate.dt.year
```


```python
# Drop date information because we only care the year
group.drop('TransactionDate',axis=1,inplace=True)
```


```python
group.head(10)
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
      <th>CustomerID</th>
      <th>Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
</div>



**Now, the group dataframe is what we want**

## Step 3. Calculate cumulative transaction amounts


```python
# Merge customer_new and group so that each custoemr has its group info
transaction = pd.merge(customer_new,
                       group,
                       on = 'CustomerID')
```


```python
# extract year of each transation
transaction['year'] = transaction.TransactionDate.dt.year
```


```python
# pre is a dataframe grouped by group and year(transaction) and each transaction age
pre=transaction.groupby(['Group','year']).Amount.sum().groupby(level=[0]).cumsum().reset_index()
pre['Origin']=12*(pre['year']-pre['Group']+1)
#Amount_cmltv=pd.DataFrame(transaction.groupby(['Group','year']).Amount.sum().groupby(level=[0]).cumsum().unstack().to_records())
```


```python
pre.head()
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
      <th>Group</th>
      <th>year</th>
      <th>Amount</th>
      <th>Origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>2010</td>
      <td>2259.67</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010</td>
      <td>2011</td>
      <td>3614.78</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>2012</td>
      <td>5274.81</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>2013</td>
      <td>6632.37</td>
      <td>48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>2014</td>
      <td>7930.69</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Conver the df above to pivot table we want using crosstab function
Amount_cmltv=pd.crosstab(pre.Group,pre.Origin,values=pre.Amount,aggfunc=lambda x:x)
```


```python
# set the index as we want
Amount_cmltv.index=['2010-01-01 - 2010-12-31',
                      '2011-01-01 - 2011-12-31',
                      '2012-01-01 - 2012-12-31',
                      '2013-01-01 - 2013-12-31',
                      '2014-01-01 - 2014-12-31',
                      '2015-01-01 - 2015-12-31',]

#Amount_cmltv.columns = ['Origin', '12','24','36','48','60','72']

```


```python
Amount_cmltv
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
      <th>Origin</th>
      <th>12</th>
      <th>24</th>
      <th>36</th>
      <th>48</th>
      <th>60</th>
      <th>72</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01 - 2010-12-31</th>
      <td>2259.67</td>
      <td>3614.78</td>
      <td>5274.81</td>
      <td>6632.37</td>
      <td>7930.69</td>
      <td>8964.49</td>
    </tr>
    <tr>
      <th>2011-01-01 - 2011-12-31</th>
      <td>2238.46</td>
      <td>3757.90</td>
      <td>5465.99</td>
      <td>6703.11</td>
      <td>7862.24</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-01-01 - 2012-12-31</th>
      <td>2181.35</td>
      <td>3874.69</td>
      <td>5226.86</td>
      <td>6501.85</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-01 - 2013-12-31</th>
      <td>2179.85</td>
      <td>3609.81</td>
      <td>5227.75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-01-01 - 2014-12-31</th>
      <td>1830.85</td>
      <td>3262.05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-01 - 2015-12-31</th>
      <td>1912.17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Step 4. Calculate cumulative transaction amounts



```python
# Copy the df since they have the same structure
newcust_cmltv=Amount_cmltv.copy()
```

**Get # new Customers for each year**


```python
#get # new customers for each year
newcust=group.groupby('Group').CustomerID.count()
newcust
```




    Group
    2010    172
    2011    170
    2012    163
    2013    180
    2014    155
    2015    160
    Name: CustomerID, dtype: int64



**newcustomer table has the same structure with amount table instead of numbers**


```python
import numpy as np

for i in range(newcust_cmltv.shape[0]):
    newcust_cmltv.iloc [i,]=[newcust.iloc[i]]*(6-i)+[np.NaN]*i
```


```python
newcust_cmltv
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
      <th>Origin</th>
      <th>12</th>
      <th>24</th>
      <th>36</th>
      <th>48</th>
      <th>60</th>
      <th>72</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01 - 2010-12-31</th>
      <td>172.0</td>
      <td>172.0</td>
      <td>172.0</td>
      <td>172.0</td>
      <td>172.0</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>2011-01-01 - 2011-12-31</th>
      <td>170.0</td>
      <td>170.0</td>
      <td>170.0</td>
      <td>170.0</td>
      <td>170.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-01-01 - 2012-12-31</th>
      <td>163.0</td>
      <td>163.0</td>
      <td>163.0</td>
      <td>163.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-01 - 2013-12-31</th>
      <td>180.0</td>
      <td>180.0</td>
      <td>180.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-01-01 - 2014-12-31</th>
      <td>155.0</td>
      <td>155.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-01 - 2015-12-31</th>
      <td>160.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Step 6. Historic CLV


```python
H_CLV=Amount_cmltv/newcust_cmltv
```


```python
H_CLV
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
      <th>Origin</th>
      <th>12</th>
      <th>24</th>
      <th>36</th>
      <th>48</th>
      <th>60</th>
      <th>72</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01 - 2010-12-31</th>
      <td>13.137616</td>
      <td>21.016163</td>
      <td>30.667500</td>
      <td>38.560291</td>
      <td>46.108663</td>
      <td>52.119128</td>
    </tr>
    <tr>
      <th>2011-01-01 - 2011-12-31</th>
      <td>13.167412</td>
      <td>22.105294</td>
      <td>32.152882</td>
      <td>39.430059</td>
      <td>46.248471</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-01-01 - 2012-12-31</th>
      <td>13.382515</td>
      <td>23.771104</td>
      <td>32.066626</td>
      <td>39.888650</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-01 - 2013-12-31</th>
      <td>12.110278</td>
      <td>20.054500</td>
      <td>29.043056</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-01-01 - 2014-12-31</th>
      <td>11.811935</td>
      <td>21.045484</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-01 - 2015-12-31</th>
      <td>11.951062</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib.ticker import FormatStrFormatter
plt.figure(figsize=(20,10))
plt.plot(H_CLV.T,linewidth=5,marker='o')
plt.title('Historic CLV')
plt.ylabel('HistoricCLV')
plt.xlabel('Age')
plt.legend(list(Amount_cmltv.index))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%d '))
plt.grid()
```


![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/clv.png)



```python
newcust_cmltv
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
      <th>Origin</th>
      <th>12</th>
      <th>24</th>
      <th>36</th>
      <th>48</th>
      <th>60</th>
      <th>72</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01 - 2010-12-31</th>
      <td>172.0</td>
      <td>172.0</td>
      <td>172.0</td>
      <td>172.0</td>
      <td>172.0</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>2011-01-01 - 2011-12-31</th>
      <td>170.0</td>
      <td>170.0</td>
      <td>170.0</td>
      <td>170.0</td>
      <td>170.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-01-01 - 2012-12-31</th>
      <td>163.0</td>
      <td>163.0</td>
      <td>163.0</td>
      <td>163.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-01 - 2013-12-31</th>
      <td>180.0</td>
      <td>180.0</td>
      <td>180.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-01-01 - 2014-12-31</th>
      <td>155.0</td>
      <td>155.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-01 - 2015-12-31</th>
      <td>160.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
H_CLV
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
      <th>Origin</th>
      <th>12</th>
      <th>24</th>
      <th>36</th>
      <th>48</th>
      <th>60</th>
      <th>72</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01 - 2010-12-31</th>
      <td>13.137616</td>
      <td>21.016163</td>
      <td>30.667500</td>
      <td>38.560291</td>
      <td>46.108663</td>
      <td>52.119128</td>
    </tr>
    <tr>
      <th>2011-01-01 - 2011-12-31</th>
      <td>13.167412</td>
      <td>22.105294</td>
      <td>32.152882</td>
      <td>39.430059</td>
      <td>46.248471</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-01-01 - 2012-12-31</th>
      <td>13.382515</td>
      <td>23.771104</td>
      <td>32.066626</td>
      <td>39.888650</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-01 - 2013-12-31</th>
      <td>12.110278</td>
      <td>20.054500</td>
      <td>29.043056</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-01-01 - 2014-12-31</th>
      <td>11.811935</td>
      <td>21.045484</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-01 - 2015-12-31</th>
      <td>11.951062</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
newcust_cmltv.fillna(0)*H_CLV.fillna(0)
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
      <th>Origin</th>
      <th>12</th>
      <th>24</th>
      <th>36</th>
      <th>48</th>
      <th>60</th>
      <th>72</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01 - 2010-12-31</th>
      <td>2259.67</td>
      <td>3614.78</td>
      <td>5274.81</td>
      <td>6632.37</td>
      <td>7930.69</td>
      <td>8964.49</td>
    </tr>
    <tr>
      <th>2011-01-01 - 2011-12-31</th>
      <td>2238.46</td>
      <td>3757.90</td>
      <td>5465.99</td>
      <td>6703.11</td>
      <td>7862.24</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2012-01-01 - 2012-12-31</th>
      <td>2181.35</td>
      <td>3874.69</td>
      <td>5226.86</td>
      <td>6501.85</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2013-01-01 - 2013-12-31</th>
      <td>2179.85</td>
      <td>3609.81</td>
      <td>5227.75</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2014-01-01 - 2014-12-31</th>
      <td>1830.85</td>
      <td>3262.05</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2015-01-01 - 2015-12-31</th>
      <td>1912.17</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



**volume-weighted average of the Historic CLV for each group at each Age**


```python
newcust_cmltv.fillna(0).sum()
```




    Origin
    12    1000.0
    24     840.0
    36     685.0
    48     505.0
    60     342.0
    72     172.0
    dtype: float64




```python
# Weighted average: sum(H_CLV*newcust_cmltv)/sum(newcust_cmltv)
singleCLV=pd.DataFrame(np.sum(newcust_cmltv.fillna(0)*H_CLV.fillna(0))/np.sum(newcust_cmltv)).reset_index()
singleCLV.columns= ['Age','HistoricCLV']
```


```python
singleCLV
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
      <th>Age</th>
      <th>HistoricCLV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>12.602350</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>21.570512</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36</td>
      <td>30.942204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>39.281842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>46.178158</td>
    </tr>
    <tr>
      <th>5</th>
      <td>72</td>
      <td>52.119128</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusion

Pandas is really strong to manipulate data. However, groupby function can be very confusing when it creates multi-index or index-dataframe. When we apply sum() or crosstab() later, it is easy to get trouble. Therefore, every step to generate a dataframe worths time to scrutinize. 

