
# Groupby, Agg and Transform Tutorial

Groupby, agreegation and transform function is frequently used in data analysis with Pandas. Let's see their application with some examples.


```python
import pandas as pd
sales = pd.read_excel('D:/lj2/Machine-Learning/Exploratory Analysis/sales_transactions.xlsx')
sales.head()
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
      <th>account</th>
      <th>name</th>
      <th>order</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>10001</td>
      <td>B1-20000</td>
      <td>7</td>
      <td>33.69</td>
      <td>235.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>10001</td>
      <td>S1-27722</td>
      <td>11</td>
      <td>21.12</td>
      <td>232.32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>10001</td>
      <td>B1-86481</td>
      <td>3</td>
      <td>35.99</td>
      <td>107.97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>10005</td>
      <td>S1-06532</td>
      <td>48</td>
      <td>55.82</td>
      <td>2679.36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>10005</td>
      <td>S1-82801</td>
      <td>21</td>
      <td>13.62</td>
      <td>286.02</td>
    </tr>
  </tbody>
</table>
</div>



## Task 1. Group by get sum price for each order

[Reference](http://pbpython.com/pandas_transform.html)


```python
sales.groupby('order')['ext price'].sum()
```




    order
    10001     576.12
    10005    8185.49
    10006    3724.49
    Name: ext price, dtype: float64



## Task 2. Calculate proportion of cost of each item in each order

transform() function is used


```python
sales['TotalPrice']=sales.groupby('order')['ext price'].transform(sum)
sales['Portion']=sales['ext price']/sales['TotalPrice']
sales['Portion']=sales['Portion'].apply(lambda x: str(round(x*100,3))+"%")
sales
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
      <th>account</th>
      <th>name</th>
      <th>order</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>TotalPrice</th>
      <th>Portion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>10001</td>
      <td>B1-20000</td>
      <td>7</td>
      <td>33.69</td>
      <td>235.83</td>
      <td>576.12</td>
      <td>40.934%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>10001</td>
      <td>S1-27722</td>
      <td>11</td>
      <td>21.12</td>
      <td>232.32</td>
      <td>576.12</td>
      <td>40.325%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>383080</td>
      <td>Will LLC</td>
      <td>10001</td>
      <td>B1-86481</td>
      <td>3</td>
      <td>35.99</td>
      <td>107.97</td>
      <td>576.12</td>
      <td>18.741%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>10005</td>
      <td>S1-06532</td>
      <td>48</td>
      <td>55.82</td>
      <td>2679.36</td>
      <td>8185.49</td>
      <td>32.733%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>10005</td>
      <td>S1-82801</td>
      <td>21</td>
      <td>13.62</td>
      <td>286.02</td>
      <td>8185.49</td>
      <td>3.494%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>10005</td>
      <td>S1-06532</td>
      <td>9</td>
      <td>92.55</td>
      <td>832.95</td>
      <td>8185.49</td>
      <td>10.176%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>10005</td>
      <td>S1-47412</td>
      <td>44</td>
      <td>78.91</td>
      <td>3472.04</td>
      <td>8185.49</td>
      <td>42.417%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>10005</td>
      <td>S1-27722</td>
      <td>36</td>
      <td>25.42</td>
      <td>915.12</td>
      <td>8185.49</td>
      <td>11.18%</td>
    </tr>
    <tr>
      <th>8</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>10006</td>
      <td>S1-27722</td>
      <td>32</td>
      <td>95.66</td>
      <td>3061.12</td>
      <td>3724.49</td>
      <td>82.189%</td>
    </tr>
    <tr>
      <th>9</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>10006</td>
      <td>B1-33087</td>
      <td>23</td>
      <td>22.55</td>
      <td>518.65</td>
      <td>3724.49</td>
      <td>13.925%</td>
    </tr>
    <tr>
      <th>10</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>10006</td>
      <td>B1-33364</td>
      <td>3</td>
      <td>72.30</td>
      <td>216.90</td>
      <td>3724.49</td>
      <td>5.824%</td>
    </tr>
    <tr>
      <th>11</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>10006</td>
      <td>B1-20000</td>
      <td>-1</td>
      <td>72.18</td>
      <td>-72.18</td>
      <td>3724.49</td>
      <td>-1.938%</td>
    </tr>
  </tbody>
</table>
</div>



## Task 3. Multi-index groupby 


```python
Item    Price  Minimum Most_Common_Price
0 Coffee  1      1       2
1 Coffee  2      1       2
2 Coffee  2      1       2
3 Tea     3      3       4
4 Tea     4      3       4
5 Tea     4      3       4
```


```python
item = pd.DataFrame({'Item':['Coffee','Coffee','Coffee','Coffee','Coffee','Coffee','Coffee','Coffee',
                             'Tea','Tea','Tea','Tea','Tea','Tea','Tea'],
                     'Brand':['Star','Star','Moon','Moon','Star','Moon','Star','Moon',
                              'Garden','Garden','Park','Garden','Garden','Park','Park'],
                     'Price':[1,1,2,2,3,3,3,4,4,5,5,5,5,6,6]})
```


```python
item
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
      <th>Brand</th>
      <th>Item</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Park</td>
      <td>Tea</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Park</td>
      <td>Tea</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Park</td>
      <td>Tea</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
item.groupby('Item')['Brand'].value_counts()
```




    Item    Brand 
    Coffee  Moon      4
            Star      4
    Tea     Garden    4
            Park      3
    Name: Brand, dtype: int64



It is a multi-index series

### Task 3.1. Multi-index groupby with relative portion

[Python Data Science](https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html)


Method 1.


```python
item.groupby('Item')['Brand'].value_counts()/item.groupby('Item')['Brand'].count()
```




    Item    Brand 
    Coffee  Moon      0.500000
            Star      0.500000
    Tea     Garden    0.571429
            Park      0.428571
    Name: Brand, dtype: float64



Method 2 (works better for more complex dataframe).


```python
grouper = item.groupby('Item')['Brand'].value_counts()
grouper/grouper.groupby(level=[0]).transform(sum)
```




    Item    Brand 
    Coffee  Moon      0.500000
            Star      0.500000
    Tea     Garden    0.571429
            Park      0.428571
    Name: Brand, dtype: float64



### Task 3.2. Get item in multi-index series

Grouper is a special dataframe. Let's study it.


```python
grouper
```




    Item    Brand 
    Coffee  Moon      4
            Star      4
    Tea     Garden    4
            Park      3
    Name: Brand, dtype: int64




```python
grouper.Coffee
```




    Brand
    Moon    4
    Star    4
    Name: Brand, dtype: int64




```python
grouper['Coffee','Moon']
```




    4




```python
## create a pivot table
grouper.unstack()
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
      <th>Brand</th>
      <th>Garden</th>
      <th>Moon</th>
      <th>Park</th>
      <th>Star</th>
    </tr>
    <tr>
      <th>Item</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Coffee</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Tea</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouper.unstack().stack()
```




    Item    Brand 
    Coffee  Moon      4.0
            Star      4.0
    Tea     Garden    4.0
            Park      3.0
    dtype: float64



### Task 3.3. More levels get proportions


```python
# add one more column
item['Flavor']=['Black','Black','Black','Latte','Black','Latte','Black','Latte',
                 'Fruit','Milk','Fruit','Milk','Milk','Milk','Fruit']
```


```python
item
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
      <th>Brand</th>
      <th>Item</th>
      <th>Price</th>
      <th>Flavor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>1</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>1</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>2</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>2</td>
      <td>Latte</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>3</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>3</td>
      <td>Latte</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>3</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>4</td>
      <td>Latte</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>4</td>
      <td>Fruit</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
      <td>Milk</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Park</td>
      <td>Tea</td>
      <td>5</td>
      <td>Fruit</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
      <td>Milk</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
      <td>Milk</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Park</td>
      <td>Tea</td>
      <td>6</td>
      <td>Milk</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Park</td>
      <td>Tea</td>
      <td>6</td>
      <td>Fruit</td>
    </tr>
  </tbody>
</table>
</div>




```python
item.groupby(['Item','Flavor'])['Brand'].value_counts()
```




    Item    Flavor  Brand 
    Coffee  Black   Star      4
                    Moon      1
            Latte   Moon      3
    Tea     Fruit   Park      2
                    Garden    1
            Milk    Garden    3
                    Park      1
    Name: Brand, dtype: int64




```python
item.groupby(['Item','Flavor'])['Brand'].count()
```




    Item    Flavor
    Coffee  Black     5
            Latte     3
    Tea     Fruit     3
            Milk      4
    Name: Brand, dtype: int64



#### Task 3.3.1 We still need to calculate the proportion of each group


```python
# it will give us a problem
#item.groupby(['Item','Flavor'])['Brand'].value_counts()/item.groupby(['Item','Flavor'])['Brand'].count()

grouper1=item.groupby(['Item','Flavor'])['Brand'].value_counts()
grouper1/grouper1.groupby(level=[0,1]).transform(sum)
```




    Item    Flavor  Brand 
    Coffee  Black   Star      0.800000
                    Moon      0.200000
            Latte   Moon      1.000000
    Tea     Fruit   Park      0.666667
                    Garden    0.333333
            Milk    Garden    0.750000
                    Park      0.250000
    Name: Brand, dtype: float64



#### Task 3.3.2. get most frequent combination for each item.


```python
grouper1
```




    Item    Flavor  Brand 
    Coffee  Black   Star      4
                    Moon      1
            Latte   Moon      3
    Tea     Fruit   Park      2
                    Garden    1
            Milk    Garden    3
                    Park      1
    Name: Brand, dtype: int64




```python
grouper1[grouper1==grouper1.groupby(level=[0,1]).transform(max)]
```




    Item    Flavor  Brand 
    Coffee  Black   Star      4
            Latte   Moon      3
    Tea     Fruit   Park      2
            Milk    Garden    3
    Name: Brand, dtype: int64



#### Task 3.3.3 get the most frequent price

Inspired by [Stackflow problem](https://stackoverflow.com/questions/47898768/how-to-groupby-transform-to-value-counts-in-pandas)


```python
grouper2=item.groupby((['Item','Brand'])).Price
```


```python
grouper2.value_counts()
```




    Item    Brand   Price
    Coffee  Moon    2        2
                    3        1
                    4        1
            Star    1        2
                    3        2
    Tea     Garden  5        3
                    4        1
            Park    6        2
                    5        1
    Name: Price, dtype: int64




```python
item['Most Frequent']=grouper2.transform(lambda x: x.mode()[0])
```


```python
item
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
      <th>Brand</th>
      <th>Item</th>
      <th>Price</th>
      <th>Flavor</th>
      <th>Most Frequent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>1</td>
      <td>Black</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>1</td>
      <td>Black</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>2</td>
      <td>Black</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>2</td>
      <td>Latte</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>3</td>
      <td>Black</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>3</td>
      <td>Latte</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Star</td>
      <td>Coffee</td>
      <td>3</td>
      <td>Black</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Moon</td>
      <td>Coffee</td>
      <td>4</td>
      <td>Latte</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>4</td>
      <td>Fruit</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
      <td>Milk</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Park</td>
      <td>Tea</td>
      <td>5</td>
      <td>Fruit</td>
      <td>6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
      <td>Milk</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Garden</td>
      <td>Tea</td>
      <td>5</td>
      <td>Milk</td>
      <td>5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Park</td>
      <td>Tea</td>
      <td>6</td>
      <td>Milk</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Park</td>
      <td>Tea</td>
      <td>6</td>
      <td>Fruit</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### For future study

[Python Data Science](https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html)
