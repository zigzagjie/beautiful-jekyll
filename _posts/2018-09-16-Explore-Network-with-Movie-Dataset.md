---
layout: post
title: Explore Network with Movie Dataset
subtitle: Visualization with NetworkX in Python
gh-repo: zigzagjie/Data-Science
gh-badge: [star, fork, follow]
tags: [Pandas,Visualization]
---

# Association Rules applied in Movie database

This notebook walks through some fun analysis on The Movie Database (TMDb). The dataset is available on kaggle dataset. You can find the here [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata/home). 


Association rules analysis is frequently found in market reserach. In this dataset, association rule is applied to find the relationship of different **genres** for movies. In the movie dataset, generes for every movie is provided. 

The second application is to find the **cooperation of actors and actresses**. We would like to find pattern beween movie cooperations. 

All the results are visualized by networkx package. The function was written when I did internship at Autodesk.The drawnetwork function is just normal. 

**Motivation**

I love network. A lot of interactions happen around us everyday and it is really cool to visualiza them by graph.

Ok. Let's get started! Have fun!

## Part 0. Prepare


```python
import pandas as pd
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
```


```python
credit=pd.read_csv('tmdb_5000_credits.csv')
movie=pd.read_csv('tmdb_5000_movies.csv')

movie.drop(['homepage','tagline'],axis=1,inplace=True)
movie.dropna(inplace=True)
```

## Part 1. Genres Analysis


```python
movie.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4799 entries, 0 to 4802
    Data columns (total 18 columns):
    budget                  4799 non-null int64
    genres                  4799 non-null object
    id                      4799 non-null int64
    keywords                4799 non-null object
    original_language       4799 non-null object
    original_title          4799 non-null object
    overview                4799 non-null object
    popularity              4799 non-null float64
    production_companies    4799 non-null object
    production_countries    4799 non-null object
    release_date            4799 non-null object
    revenue                 4799 non-null int64
    runtime                 4799 non-null float64
    spoken_languages        4799 non-null object
    status                  4799 non-null object
    title                   4799 non-null object
    vote_average            4799 non-null float64
    vote_count              4799 non-null int64
    dtypes: float64(3), int64(4), object(11)
    memory usage: 712.4+ KB
    

### Data Manipulation

- Read genres in json format and convert to the data structure that Python can handle
- Append all the genres in one list for each movie


```python
import json
movie['genres']=movie.genres.apply(lambda x: json.loads(x))

def convertList(inputList):
    ge=[]
    for dic in inputList:
        ge.append(dic['name'])
    return ge

def getFirst(inputList):
    if len(inputList)==0:
        return np.NaN
    else:
        return inputList[0]
```


```python
movie.genres.apply(lambda x: convertList(x)).head()
```

- Remove movies with empty genre list
- Convert columns with list to several columns with binary info

Ex: for the first movie, its genres is [Action, Adventure, Fantasy, Science Fiction], then we create three corresponding columns to represent them and set its value as 1.


```python
g=pd.DataFrame(movie.genres.apply(lambda x: convertList(x)))
g=g[g.genres.apply(len)!=0]
```


```python
for index, row in g.iterrows():
    for item in row['genres']:
        g.at[index,item]=1
g=g.fillna(0)
```


```python
g.head()
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
      <th>genres</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Fantasy</th>
      <th>Science Fiction</th>
      <th>Crime</th>
      <th>Drama</th>
      <th>Thriller</th>
      <th>Animation</th>
      <th>Family</th>
      <th>Western</th>
      <th>Comedy</th>
      <th>Romance</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>History</th>
      <th>War</th>
      <th>Music</th>
      <th>Documentary</th>
      <th>Foreign</th>
      <th>TV Movie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Action, Adventure, Fantasy, Science Fiction]</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Adventure, Fantasy, Action]</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Action, Adventure, Crime]</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Action, Crime, Drama, Thriller]</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Action, Adventure, Science Fiction]</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Apply Assciation rules

The details of association rules is skipped. 


```python
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(g.drop('genres',axis=1), min_support=0.02, use_colnames=True)
```


```python
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules=rules[(rules['antecedents'].apply(len)==1)&(rules['consequents'].apply(len)==1)]
rules.sort_values('confidence',ascending=False).head()
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
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>(History)</td>
      <td>(Drama)</td>
      <td>0.041282</td>
      <td>0.481140</td>
      <td>0.036672</td>
      <td>0.888325</td>
      <td>1.846292</td>
      <td>0.016810</td>
      <td>4.646156</td>
    </tr>
    <tr>
      <th>44</th>
      <td>(Animation)</td>
      <td>(Family)</td>
      <td>0.049036</td>
      <td>0.107502</td>
      <td>0.040863</td>
      <td>0.833333</td>
      <td>7.751787</td>
      <td>0.035592</td>
      <td>5.354987</td>
    </tr>
    <tr>
      <th>36</th>
      <td>(War)</td>
      <td>(Drama)</td>
      <td>0.030176</td>
      <td>0.481140</td>
      <td>0.024728</td>
      <td>0.819444</td>
      <td>1.703131</td>
      <td>0.010209</td>
      <td>2.873686</td>
    </tr>
    <tr>
      <th>42</th>
      <td>(Mystery)</td>
      <td>(Thriller)</td>
      <td>0.072925</td>
      <td>0.266974</td>
      <td>0.050712</td>
      <td>0.695402</td>
      <td>2.604756</td>
      <td>0.031243</td>
      <td>2.406538</td>
    </tr>
    <tr>
      <th>31</th>
      <td>(Romance)</td>
      <td>(Drama)</td>
      <td>0.187343</td>
      <td>0.481140</td>
      <td>0.126362</td>
      <td>0.674497</td>
      <td>1.401872</td>
      <td>0.036224</td>
      <td>1.594024</td>
    </tr>
  </tbody>
</table>
</div>



**Important interpretation**

It is ok if you do not understand association rules. I can explain the results above in very plain language. 

The rules dataset can be translated as : 

   - history genre movie has 0.041 probability(antecedent support) to appear and drama genre movie has 0.48 probability(consequent support) to appear. Moreover, when the movie is history type, it has 88.83% probability(confidence) to be a drama movie as well. That should make sense. Other rules can be interpreted in a similary way.
   - The higher the confidence of the rules, the closer relationship between two genres.

### NetworkX Visualizes the rules


```python
import networkx as nx

def drawNetwork(ant2):
    G1 = nx.DiGraph()

    for index, row in ant2.iterrows():
        #add node
        G1.add_node(list(row['antecedents'])[0],weight=round(row['antecedent support'],3))
        #add node
        G1.add_node(list(row['consequents'])[0],weight=round(row['consequent support'],3))
        #add edge
        G1.add_edge(list(row['antecedents'])[0],list(row['consequents'])[0],
                   weight=round(row['confidence'],3))
    #G=nx.from_pandas_edgelist(ant2, 'antecedants', 'consequents', ['confidence'])
    f, ax = plt.subplots(figsize=(20,20))

    #plt.figure(figsize=(20,20))
    pos = nx.spring_layout(G1)
    #nx.draw_networkx_edges(G1, pos, arrows=True)
    #nx.draw(G1,pos=pos,with_labels = True,arrows=True)
    #nx.draw_networkx_edges(G1,pos)
    edges=G1.edges()
    #colors = [G[u][v]['color'] for u,v in edges]
    #weights = [G[u][v]['weight'] for u,v in edges]
    labels = nx.get_edge_attributes(G1,'weight')
    node_weight=[150*nx.get_node_attributes(G1,'weight')[key] for key in G1.nodes]
    val_map=nx.get_node_attributes(G1,'weight')
    values= [10000*val_map.get(node, 0.25) for node in G1.nodes()]

    #nx.draw(G1, cmap=plt.get_cmap('jet'), node_color=values)
    nx.draw_networkx_nodes(G1, pos, cmap=plt.get_cmap('jet'),
                           node_size = values,node_color='orange',alpha=0.6,ax=ax)
    nx.draw_networkx_labels(G1, pos,ax=ax,fontsize=14)
    #nx.draw_networkx_edges(G1, pos, edgelist=G1.edges(), arrows=True)
    nx.draw_networkx_edges(G1, pos,edgelist=edges, edge_color='lightskyblue', arrows=True,ax=ax)
    #nx.draw_networkx_edge_labels(G1,pos,edge_labels=labels,ax=ax)

    #sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'))
    #sm._A = []
    #plt.colorbar(sm)
    return f
```


```python
drawNetwork(rules);
```


![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/network1.png)


### Conclusion

1. drama, comedy, thriller, action seem to be top four popuolar genres in movie history (by their size/antecedent support/confidence support). They can also be interpreted as a general type because almost all the edges are inward. We can also find the types connected is more specific. 
2. For the family movie, it can also be adventure, animation and comedy. Apparantly, thriller movies are not suitable for a family. 
3. The result is reasonable based on our life experience

to be continued...

## Part 2. Actor/Actresses

From Credit dataset, we can obtain all the names of actresses and actors for each movie. We decide to extract them to see the preference of cooperation of each actor/actresses.

### Data Manipulation and Rules are similarly implemented


```python
credit['cast']=credit.cast.apply(lambda x: json.loads(x))

def convertList(inputList):
    ge=[]
    for dic in inputList:
        ge.append(dic['name'])
    return ge

def getFirst(inputList):
    if len(inputList)==0:
        return np.NaN
    else:
        return inputList[0]
```


```python
cast=pd.DataFrame(credit.cast.apply(lambda x: convertList(x)))
cast=cast[cast.cast.apply(len)!=0]
```


```python
cast.head()
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
      <th>cast</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Sam Worthington, Zoe Saldana, Sigourney Weave...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Johnny Depp, Orlando Bloom, Keira Knightley, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Daniel Craig, Christoph Waltz, Léa Seydoux, R...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Christian Bale, Michael Caine, Gary Oldman, A...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Taylor Kitsch, Lynn Collins, Samantha Morton,...</td>
    </tr>
  </tbody>
</table>
</div>




```python
for index, row in cast.iterrows():
    i=0
    for item in row['cast']:
        if i<5:
            cast.at[index,item]=1
        else:
            break
        i=i+1
cast=cast.fillna(0)
```

the conversion takes a longer time to process due to too many new columns to create. The cast dataframe will be high-dimensional. We early stop the rules serach.


```python
cast.shape
```




    (4760, 9391)




```python
frequent_itemsets = apriori(cast.drop('cast',axis=1), min_support=0.0001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules=rules[(rules['antecedents'].apply(len)==1)&(rules['consequents'].apply(len)==1)]
rules.sort_values('confidence',ascending=False).head()
```


**Early Stop Here to save time**



```python
rules.sort_values('confidence',ascending=False).head()
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
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>(James Doohan)</td>
      <td>(DeForest Kelley)</td>
      <td>0.001471</td>
      <td>0.001471</td>
      <td>0.001471</td>
      <td>1.000000</td>
      <td>680.000000</td>
      <td>0.001468</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>34</th>
      <td>(DeForest Kelley)</td>
      <td>(James Doohan)</td>
      <td>0.001471</td>
      <td>0.001471</td>
      <td>0.001471</td>
      <td>1.000000</td>
      <td>680.000000</td>
      <td>0.001468</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(George Takei)</td>
      <td>(Leonard Nimoy)</td>
      <td>0.001471</td>
      <td>0.001681</td>
      <td>0.001261</td>
      <td>0.857143</td>
      <td>510.000000</td>
      <td>0.001258</td>
      <td>6.988235</td>
    </tr>
    <tr>
      <th>33</th>
      <td>(James Doohan)</td>
      <td>(George Takei)</td>
      <td>0.001471</td>
      <td>0.001471</td>
      <td>0.001261</td>
      <td>0.857143</td>
      <td>582.857143</td>
      <td>0.001258</td>
      <td>6.989706</td>
    </tr>
    <tr>
      <th>32</th>
      <td>(George Takei)</td>
      <td>(James Doohan)</td>
      <td>0.001471</td>
      <td>0.001471</td>
      <td>0.001261</td>
      <td>0.857143</td>
      <td>582.857143</td>
      <td>0.001258</td>
      <td>6.989706</td>
    </tr>
  </tbody>
</table>
</div>




```python
drawNetwork(rules);
```


![png](https://raw.githubusercontent.com/zigzagjie/jieloudata/master/img/blog/network2.png)


Seems like the association rules do not work well here. 

## Conclusion

1. It would be better to join two tables at the very beginning. We would do it next time
2. Analysis beyong network can be various too. Let's expect the next time analysis.
