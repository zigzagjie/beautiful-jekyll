---
layout: post
title: Clustering Grocery Items
subtitle: Winter projects 2
gh-repo: zigzagjie/Data-Science
gh-badge: [star, fork, follow]
tags: [Pandas,Manipulation,WinterProject]
---
# Clustering Grocery Items

## Project Description

This project is the 9th problem at Take-Home Data Science Challenge.

Company XYZ is an online grocery store. In the current version of the website, they have manually grouped the items into a few categories based on their experience. However, they now have a lot of data about user purchase history. Therefore, they would like to put the data into use! 

This is what they asked you to do: 

- The company founder wants to meet with some of the best customers to go through a focus group with them. You are asked to send the ID of the following customers to the founder: 
   - the customer who bought the most items overall in her lifetime 
   - for each item the customer who bought that product the most 
-    Cluster items based on user co-purchase history. That is, create clusters of products that have the highest probability of being bought together. The goal of this is to replace the old/manually created categories with these new ones. Each item can belong to just one cluster.


## Data Loading


```
import pandas as pd
```


```
purchase=pd.read_csv("grocery/purchase_history.csv")
items = pd.read_csv("grocery/item_to_id.csv")
```


```
purchase.head()
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
      <th>user_id</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>222087</td>
      <td>27,26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1343649</td>
      <td>6,47,17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>404134</td>
      <td>18,12,23,22,27,43,38,20,35,1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1110200</td>
      <td>9,23,2,20,26,47,37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>224107</td>
      <td>31,18,5,13,1,21,48,16,26,2,44,32,20,37,42,35,4...</td>
    </tr>
  </tbody>
</table>
</div>



## Data Manipulation

The dataset we have right now it eh user_id with the ids of items they bought. However, for each user, its item record is aggregated into strings. So first, we need extract info from it. The best way is to generate a table with each user and each product.

I have detailed the related techniques in the last blog. [Long to Wide](https://zigzagjie.github.io/2018-12-28-Long-to-Wide-in-Pandas/)

If you cannot understand why I did it, please go the last post. 


```
unique_items = set([int(item) for rows in purchase.id.str.split(",") for item in rows])
items_dict = dict.fromkeys(unique_items,0)
```


```
def frequency(df):
    ids = df.id.str.split(",").sum()
    items_dict = dict.fromkeys(unique_items,0)
    for i in ids:
        items_dict[int(i)] = items_dict.get(int(i),0)+1
    return pd.Series(items_dict)

users = purchase.groupby("user_id").apply(frequency)
users.head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
    </tr>
    <tr>
      <th>user_id</th>
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
      <th>47</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>223</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>




```
users.columns.name = "Items"
users.head()
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
      <th>Items</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
    </tr>
    <tr>
      <th>user_id</th>
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
      <th>47</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>223</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



### Which customer buy most?


```
users.sum(axis=1).sort_values(ascending=False).head()
```




    user_id
    269335    72
    367872    70
    599172    64
    397623    64
    377284    63
    dtype: int64



**Customer 269335 buys the most**

### For each item, which customer buy most?


```
most_purchases = pd.concat([users.T.max(axis=1),users.T.idxmax(axis=1)],axis=1)
most_purchases.columns = ['Count','UserID']
most_purchases
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
      <th>Count</th>
      <th>UserID</th>
    </tr>
    <tr>
      <th>Items</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>31625</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>31625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>154960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>5289</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>217277</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>334664</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>175865</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>151926</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>269335</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>618914</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>367872</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3</td>
      <td>557904</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4</td>
      <td>653800</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3</td>
      <td>172120</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3</td>
      <td>143741</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3</td>
      <td>73071</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4</td>
      <td>366155</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5</td>
      <td>917199</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3</td>
      <td>31625</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4</td>
      <td>885474</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4</td>
      <td>884172</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
      <td>1199670</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5</td>
      <td>920002</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>189913</td>
    </tr>
    <tr>
      <th>25</th>
      <td>4</td>
      <td>68282</td>
    </tr>
    <tr>
      <th>26</th>
      <td>4</td>
      <td>967573</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4</td>
      <td>956666</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4</td>
      <td>204624</td>
    </tr>
    <tr>
      <th>29</th>
      <td>4</td>
      <td>394348</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>21779</td>
    </tr>
    <tr>
      <th>31</th>
      <td>3</td>
      <td>289360</td>
    </tr>
    <tr>
      <th>32</th>
      <td>4</td>
      <td>109578</td>
    </tr>
    <tr>
      <th>33</th>
      <td>3</td>
      <td>1310207</td>
    </tr>
    <tr>
      <th>34</th>
      <td>4</td>
      <td>305916</td>
    </tr>
    <tr>
      <th>35</th>
      <td>3</td>
      <td>450482</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4</td>
      <td>269335</td>
    </tr>
    <tr>
      <th>37</th>
      <td>4</td>
      <td>46757</td>
    </tr>
    <tr>
      <th>38</th>
      <td>4</td>
      <td>255546</td>
    </tr>
    <tr>
      <th>39</th>
      <td>5</td>
      <td>599172</td>
    </tr>
    <tr>
      <th>40</th>
      <td>4</td>
      <td>38872</td>
    </tr>
    <tr>
      <th>41</th>
      <td>4</td>
      <td>133355</td>
    </tr>
    <tr>
      <th>42</th>
      <td>4</td>
      <td>80215</td>
    </tr>
    <tr>
      <th>43</th>
      <td>4</td>
      <td>996380</td>
    </tr>
    <tr>
      <th>44</th>
      <td>4</td>
      <td>31625</td>
    </tr>
    <tr>
      <th>45</th>
      <td>5</td>
      <td>1198106</td>
    </tr>
    <tr>
      <th>46</th>
      <td>4</td>
      <td>1218645</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4</td>
      <td>384935</td>
    </tr>
    <tr>
      <th>48</th>
      <td>3</td>
      <td>335841</td>
    </tr>
  </tbody>
</table>
</div>



## First apply association rules

To find product-wise probability to buy together


```
binary = users.applymap(lambda x:0 if x==0 else 1)
```


```
binary.head()
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
      <th>Items</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
    </tr>
    <tr>
      <th>user_id</th>
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
      <th>47</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>223</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



**Another association rules applied project goes here: [click me](https://zigzagjie.github.io/2018-09-16-Explore-Network-with-Movie-Dataset/)**


```
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(binary, min_support=0.02, use_colnames=True)

```


```
rules = association_rules(frequent_itemsets, support_only=True,min_threshold=0.01)

rules = rules[["antecedents","consequents","support"]]
rules=rules[(rules['antecedents'].apply(len)==1)&(rules['consequents'].apply(len)==1)]

rules.antecedents=rules.antecedents.apply(lambda x: [item for item in x][0])
rules.consequents=rules.consequents.apply(lambda x: [item for item in x][0])
```


```
rules.support = rules.support.apply(lambda x: 1-x)
rules.head()
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.802090</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0.802090</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0.878843</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>0.878843</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
      <td>0.940526</td>
    </tr>
  </tbody>
</table>
</div>




```
rules.support.describe()
```




    count    2182.000000
    mean        0.924455
    std         0.036978
    min         0.792968
    25%         0.905636
    50%         0.925779
    75%         0.956470
    max         0.979988
    Name: support, dtype: float64



### Using pivot table to convert to similarity matrix and try clustering

- AgglomerativeClustering is a hierachical method.
- DBSCAN is a density-based clustering method.
- Spectral Clusteirng works for the similarity matrix(affinity matrix).
- KMeans is most common used clustering method based on partition.


```
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

data_matrix = rules.pivot_table(index = "antecedents",columns="consequents",values = "support").fillna(0)
#model = AgglomerativeClustering(affinity='precomputed', n_clusters=10, linkage='average').fit(data_matrix)
#model = DBSCAN(metric="precomputed",eps=0.1).fit(data_matrix)
#model = SpectralClustering(n_clusters = 11,affinity='precomputed').fit(data_matrix)
model = KMeans(n_clusters = 10).fit(data_matrix)

labels = pd.DataFrame({"label":pd.Series(model.labels_),"Item_id":list(range(1,49))})
results=items.merge(labels,on='Item_id').sort_values("label")

results['Index']=results.groupby('label').cumcount()+1

results
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
      <th>Item_name</th>
      <th>Item_id</th>
      <th>label</th>
      <th>Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>eggs</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>butter</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>yogurt</td>
      <td>48</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>milk</td>
      <td>16</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cheeses</td>
      <td>21</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>41</th>
      <td>sandwich bags</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>aluminum foil</td>
      <td>15</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>39</th>
      <td>toilet paper</td>
      <td>33</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>38</th>
      <td>paper towels</td>
      <td>24</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20</th>
      <td>waffles</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>pasta</td>
      <td>31</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>coffee</td>
      <td>43</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>laundry detergent</td>
      <td>18</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>37</th>
      <td>dishwashing</td>
      <td>27</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>soda</td>
      <td>9</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>juice</td>
      <td>38</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tea</td>
      <td>23</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>28</th>
      <td>cherries</td>
      <td>25</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>cucumbers</td>
      <td>42</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33</th>
      <td>cauliflower</td>
      <td>45</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>32</th>
      <td>carrots</td>
      <td>10</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>31</th>
      <td>broccoli</td>
      <td>44</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>30</th>
      <td>apples</td>
      <td>32</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>29</th>
      <td>grapefruit</td>
      <td>20</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>35</th>
      <td>lettuce</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>27</th>
      <td>berries</td>
      <td>40</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>23</th>
      <td>poultry</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
    </tr>
    <tr>
      <th>25</th>
      <td>pork</td>
      <td>47</td>
      <td>4</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sandwich loaves</td>
      <td>39</td>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dinner rolls</td>
      <td>37</td>
      <td>4</td>
      <td>13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tortillas</td>
      <td>34</td>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bagels</td>
      <td>13</td>
      <td>4</td>
      <td>15</td>
    </tr>
    <tr>
      <th>26</th>
      <td>bananas</td>
      <td>46</td>
      <td>4</td>
      <td>16</td>
    </tr>
    <tr>
      <th>47</th>
      <td>pet items</td>
      <td>3</td>
      <td>4</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>sugar</td>
      <td>1</td>
      <td>4</td>
      <td>18</td>
    </tr>
    <tr>
      <th>24</th>
      <td>beef</td>
      <td>17</td>
      <td>4</td>
      <td>19</td>
    </tr>
    <tr>
      <th>46</th>
      <td>baby items</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cereals</td>
      <td>11</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>flour</td>
      <td>30</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ketchup</td>
      <td>41</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>spaghetti sauce</td>
      <td>26</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>canned vegetables</td>
      <td>28</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>shampoo</td>
      <td>12</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>soap</td>
      <td>35</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>44</th>
      <td>hand soap</td>
      <td>29</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>45</th>
      <td>shaving cream</td>
      <td>19</td>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>frozen vegetables</td>
      <td>22</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ice cream</td>
      <td>36</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Visualizing the clustering


```
import matplotlib.pyplot as plt

#plt.figure(figsize=(20,8))
#plt.scatter(results.label,results.Index)

fig, ax = plt.subplots(figsize=(20,8))
ax.scatter(results.label, results.Index,c=results.label)
ax.set_title("KMeans Clustering Results")
i=0
for txt in results.Item_name:
    ax.annotate(txt, (results.label.iloc[i], results.Index.iloc[i]))
    i=i+1
    
```


![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/cluster/output_24_0.png)



```
fig, ax = plt.subplots(figsize=(20,8))
ax.scatter(results.label, results.Index,c=results.label)
ax.set_title("Spectral Clustering Results")
i=0
for txt in results.Item_name:
    ax.annotate(txt, (results.label.iloc[i], results.Index.iloc[i]))
    i=i+1
```


![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/cluster/output_25_0.png)


## Conclusion

**It turns out KMeans and Spectal Clustering work better. **

We can suggest some co-purchase items to be placed nearby for customers' convenience:

1. coffe+bagels
2. broccoli+beef+berries
3. soda+dinner rolls
4. pork+cherries+cauliflower

It gives us some hints that we can put items together with their dishes. For example, people are likely to cook broccoli and beef together, then we will put them closely. Then, people will more likely to grab them together.
