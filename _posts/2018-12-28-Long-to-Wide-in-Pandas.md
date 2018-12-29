---
layout: post
title: Long to Wide Using Pandas
subtitle: Can you tackle these manipulation challenges?
gh-repo: zigzagjie/Data-Science
gh-badge: [star, fork, follow]
tags: [Pandas,Manipulation]
---
# Long to Wide in Pandas

Long table to wide table is a common task in data manipulation using Pandas. The common task could be applied by pivot table.  However, sometimes the task could be tougher. We need to use apply function with groupby object to handle it. To wrap it up, this blog is to showcase some examples.

In this blog, I would introduce a technique to manipulate some special cases with categorical data which can not be handled simply by pivot table.

## Outline

This blog consists of three tasks:

[Task1](#Task1)

[Task2](#Task2)

[Task3](#Task3)

Difficulty increases one by one.

Let's start by importing Pandas package


```python
import pandas as pd
```

<a id='Task1'></a>
## Task 1

Easy case can be solved using pivot table 

A fake dataset contains user names and food they bought.


```python
food = pd.Series(['Milk','Egg','Egg','Cake','Milk','Cake','Apple','Milk'])
usrId = pd.Series(['User1','User1','User1','User2','User2','User3','User4','User4'])

users1 = pd.DataFrame({"ID":usrId,"Food":food})[["ID","Food"]]
users1
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
      <th>ID</th>
      <th>Food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>User1</td>
      <td>Milk</td>
    </tr>
    <tr>
      <th>1</th>
      <td>User1</td>
      <td>Egg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>User1</td>
      <td>Egg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>User2</td>
      <td>Cake</td>
    </tr>
    <tr>
      <th>4</th>
      <td>User2</td>
      <td>Milk</td>
    </tr>
    <tr>
      <th>5</th>
      <td>User3</td>
      <td>Cake</td>
    </tr>
    <tr>
      <th>6</th>
      <td>User4</td>
      <td>Apple</td>
    </tr>
    <tr>
      <th>7</th>
      <td>User4</td>
      <td>Milk</td>
    </tr>
  </tbody>
</table>
</div>



**It is a typical long table with duplicate ID and different records. Our goal is to know for each user, the quantity of food they have bought. This goal is through the entire blog**

Since Food column is categorical, so we cannot use pivot table here which is more useful for multiple columns.

Solution:

- value_counts() + unstack() + fillna()


```python
users1.groupby("ID").Food.value_counts().unstack().fillna(0)
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
      <th>Food</th>
      <th>Apple</th>
      <th>Cake</th>
      <th>Egg</th>
      <th>Milk</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>User2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>User3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>User4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



<a id='Task2'></a>
## Task 2

To get the same record, dataset can be different. The food product can be aggregated already for each user. See the following dataset:


```python
food = pd.Series([["Egg","Egg","Milk"],["Cake","Milk"],["Cake"],["Apple","Milk"]])
usrId = pd.Series(['User1','User2','User3','User4'])
users2 = pd.DataFrame({"ID":usrId,"Food":food})[["ID","Food"]]
users2
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
      <th>ID</th>
      <th>Food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>User1</td>
      <td>[Egg, Egg, Milk]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>User2</td>
      <td>[Cake, Milk]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>User3</td>
      <td>[Cake]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>User4</td>
      <td>[Apple, Milk]</td>
    </tr>
  </tbody>
</table>
</div>



### Now, how to get the same summary table like above?

It will be harder. We have to use apply method enforced to the food column


```python
def convert(x):
    food_list ={}
    for item in x:
        food_list[item]=food_list.get(item,0)+1
    return pd.Series(food_list)
```

**Note: return pandas series will be converted to multiple columns in result. Remember, each row is a series also.**


```python
users2.Food.apply(convert).fillna(0).set_index(users2.ID)
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
      <th>Apple</th>
      <th>Cake</th>
      <th>Egg</th>
      <th>Milk</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>User2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>User3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>User4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



<a id='Task3'></a>
## Task 3

For the same dataset, it could be more complicated. Suppose users have aggrated products in different days. There are multiple user records with list inputs.

That's what I meant:


```python
food = pd.Series([["Egg"],["Egg","Milk"],["Cake"],["Milk"],["Cake"],["Apple","Milk"]])
usrId = pd.Series(['User1','User1','User2','User3','User3','User4'])
users3 = pd.DataFrame({"ID":usrId,"Food":food})[["ID","Food"]]
users3
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
      <th>ID</th>
      <th>Food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>User1</td>
      <td>[Egg]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>User1</td>
      <td>[Egg, Milk]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>User2</td>
      <td>[Cake]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>User3</td>
      <td>[Milk]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>User3</td>
      <td>[Cake]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>User4</td>
      <td>[Apple, Milk]</td>
    </tr>
  </tbody>
</table>
</div>



Groupby needs to be applied here added on Task2. We need to edit our convert method.

**Notice, after grouping by ID, pandas will return a pandasgroupby object to us. We need to write function to handle this object**


```python
def convertGB(df):
    ## remember groupby will give us a group of lists
    foodList=[]
    ##These two makes a big difference!
    #food_dict={"Egg":0,"Milk":0,"Cake":0,"Apple":0}
    food_dict={}
    for food in df.Food:
        foodList +=food
    #foodList is equivalent to the list we have in the Task2
    for item in foodList:
        food_dict[item]=food_dict.get(item,0)+1
    return pd.Series(food_dict)

users3.groupby("ID").apply(convertGB)
    
```




    ID          
    User1  Egg      2
           Milk     1
    User2  Cake     1
    User3  Cake     1
           Milk     1
    User4  Apple    1
           Milk     1
    dtype: int64



**Notice here!**

When we initialize food_dict with {"Egg":0,"Milk":0,"Cake":0,"Apple":0}, all the series will have the same column names so that the ideal table will print directly. Otherwise, we have to apply unstack function further. 


```python
users3.groupby("ID").apply(convertGB).unstack().fillna(0)
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
      <th>Apple</th>
      <th>Cake</th>
      <th>Egg</th>
      <th>Milk</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>User2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>User3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>User4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



That's what I meant!


```python
def convertGB(df):
    ## remember groupby will give us a group of lists
    foodList=[]
    ##These two makes a big difference!
    food_dict={"Egg":0,"Milk":0,"Cake":0,"Apple":0}
    #food_dict={}
    for food in df.Food:
        foodList +=food
    #foodList is equivalent to the list we have in the Task2
    for item in foodList:
        food_dict[item]=food_dict.get(item,0)+1
    return pd.Series(food_dict)

users3.groupby("ID").apply(convertGB)
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
      <th>Apple</th>
      <th>Cake</th>
      <th>Egg</th>
      <th>Milk</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User1</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>User2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>User3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>User4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### The ideal table pops up directly!

However, in the reality, considering lots of unique food lists, we have to get the list of unique food first to create an initial table. It require more steps.


```python
food_dict=dict.fromkeys(set([food for obs in users3.Food for food in obs]),0)
food_dict
```




    {'Apple': 0, 'Cake': 0, 'Egg': 0, 'Milk': 0}



**Now, we can create a right dictionary automatically!**

### Equivalently applied to a column


```python
def convertGB(series):
    ## remember groupby will give us a group of lists
    foodList=[]
    food_dict={}
    for food in series:
        foodList +=food
    #foodList is equivalent to the list we have in the Task2
    for item in foodList:
        food_dict[item]=food_dict.get(item,0)+1
    return pd.Series(food_dict)
    #return foodList

users3.groupby("ID").Food.apply(convertGB).unstack().fillna(0)
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
      <th>Apple</th>
      <th>Cake</th>
      <th>Egg</th>
      <th>Milk</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>User2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>User3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>User4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion

This problem is inspired by "Clustering Grocery Items" on Take-home data science challenges. Hopefully, after these 3 tasks, you can be more familiar with data manipulation in similar situation. 

What I learned from this problem:

1. For Pandas Series, index is a key and value is the value in the Pandas dictionary. In Pandas Dataframe, column is the key and each column is a series. Similarly, the index is the key and the row is a series. It is easier to understand if you think of it as a dictionary type.

2. In Groupby, the key will becomes a new index, the new aggregated result is a series. If you apply the method returning series, then it will become multi-index series or multi columns. It depends on if you have defined the multi-column series well. 

3. Apply method to a dataframe then works on each column; if applys to a column, then works on each cell.


