---
layout: post
title: Relational Dataset Manipulation
subtitle: How to merge datasets in pandas?
gh-repo: zigzagjie/Data-Science
gh-badge: [star, fork, follow]
tags: [Pandas]
---

# Relational Data

This notebook will go through merge function in pandas. We all know there are joins in SQL query. In pandas, you can achieve the same results.

This notebook is just for personal review for class 15688 [Practical Data science](http://www.datasciencecourse.org/). For the class notes, you can go to [Relational data Note](http://www.datasciencecourse.org/notes/relational_data/)

### First, let's create some fake datasets.




```python
import pandas as pd 
Person = pd.DataFrame([(1, 'Kolter', 'Zico'), 
                   (2, 'Xi', 'Edgar'), 
                   (3, 'Lee', 'Mark'), 
                   (4, 'Mani', 'Shouvik'), 
                   (5, 'Gates', 'Bill'), 
                   (6, 'Musk', 'Elon')], 
                  columns=["ID", "Last Name", "First Name"])
Person.set_index("ID", inplace=True)

Person
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
      <th>Last Name</th>
      <th>First Name</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div>




```python
Grade = pd.DataFrame([(5,'HW1',80),
                    (6,'HW1',90),
                     (6,'HW2',0),
                     (100,'HW1',100),
                     (1,'HW1',90),
                     (1,'HW2',100)],
                     columns=['PersonId','HW','Grade']
                    )
Grade.set_index("PersonId", inplace=True)

Grade
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
      <th>HW</th>
      <th>Grade</th>
    </tr>
    <tr>
      <th>PersonId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>HW1</td>
      <td>80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HW1</td>
      <td>90</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HW2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>HW1</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HW1</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HW2</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



### Description

Person dataset restores each student's informtion. ID is its primary id in the SQL context. In pandas, it is index. 

Grade dataset restores each student's grade. It can be multiple inputs for each person for they can have multiple grades. 

### Frist, we try to inner join to select all the entries both Person and Grade have.


```python
Person.reset_index()
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
      <th>Last Name</th>
      <th>First Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div>



#### Method 1: merge two dataframes by their columns


```python
pd.merge(Person.reset_index(),Grade.reset_index(),left_on = "ID", right_on="PersonId")
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
      <th>Last Name</th>
      <th>First Name</th>
      <th>PersonId</th>
      <th>HW</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Kolter</td>
      <td>Zico</td>
      <td>1</td>
      <td>HW1</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Kolter</td>
      <td>Zico</td>
      <td>1</td>
      <td>HW2</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Gates</td>
      <td>Bill</td>
      <td>5</td>
      <td>HW1</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
      <td>6</td>
      <td>HW1</td>
      <td>90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
      <td>6</td>
      <td>HW2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Method 2: merge two dataframes by their indexes


```python
pd.merge(Person, Grade, left_index=True, right_index=True)
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
      <th>Last Name</th>
      <th>First Name</th>
      <th>HW</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>HW1</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>HW2</td>
      <td>100</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
      <td>HW1</td>
      <td>80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>HW1</td>
      <td>90</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>HW2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Second, let's try left join. Get all the students with their grade. We need Person left join Grade here.

We still showcase two methods here.


```python
pd.merge(Person, Grade, how='left',left_index=True, right_index=True).rename_axis('ID')
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
      <th>Last Name</th>
      <th>First Name</th>
      <th>HW</th>
      <th>Grade</th>
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
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>HW1</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>HW2</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
      <td>HW1</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>HW1</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>HW2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(Person.reset_index(),Grade.reset_index(),how='left',left_on = "ID", right_on="PersonId").set_index('ID')
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
      <th>Last Name</th>
      <th>First Name</th>
      <th>PersonId</th>
      <th>HW</th>
      <th>Grade</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>1.0</td>
      <td>HW1</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>1.0</td>
      <td>HW2</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
      <td>5.0</td>
      <td>HW1</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>6.0</td>
      <td>HW1</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>6.0</td>
      <td>HW2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Last, let's try Outer join. Get all the students(also the ones in Grade dataset) with their grade. We need Person outer join Grade here.

We still showcase two methods here.


```python
pd.merge(Person, Grade, how='outer',left_index=True, right_index=True).rename_axis('ID')
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
      <th>Last Name</th>
      <th>First Name</th>
      <th>HW</th>
      <th>Grade</th>
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
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>HW1</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>HW2</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
      <td>HW1</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>HW1</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>HW2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>HW1</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(Person.reset_index(),Grade.reset_index(),how='outer',left_on = "ID", right_on="PersonId").set_index('ID')
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
      <th>Last Name</th>
      <th>First Name</th>
      <th>PersonId</th>
      <th>HW</th>
      <th>Grade</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>1.0</td>
      <td>HW1</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Kolter</td>
      <td>Zico</td>
      <td>1.0</td>
      <td>HW2</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>Xi</td>
      <td>Edgar</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>Lee</td>
      <td>Mark</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>Mani</td>
      <td>Shouvik</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>Gates</td>
      <td>Bill</td>
      <td>5.0</td>
      <td>HW1</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>6.0</td>
      <td>HW1</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>6.0</td>
      <td>HW2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>NaN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>HW1</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>



## End
