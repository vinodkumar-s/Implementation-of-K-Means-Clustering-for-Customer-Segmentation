# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Import KMeans and use for loop to calculate the within cluster sum of squares the data.
4. Plot the wcss for each iteration, also known as the elbow method plot.
5. Predict the clusters and plot them.

## Program:
Program to implement the K Means Clustering for Customer Segmentation.

Developed by: VINOD KUMAR S  
RegisterNumber:  212222240116
```python
import pandas as pd 
import matplotlib.pyplot as plt 
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()
data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []  
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")

```

## Output:
## data.head():

![K Means Clustering for Customer Segmentation](/Exp8_1.png)
## data.info():

![K Means Clustering for Customer Segmentation](/c2.png)
## NULL VALUES:


![K Means Clustering for Customer Segmentation](/c3.png)
## ELBOW GRAPH:

![K Means Clustering for Customer Segmentation](/c4.png)
## CLUSTER FORMATION:

![K Means Clustering for Customer Segmentation](/Exp8_5.png)
## PREDICICTED VALUE:

![K Means Clustering for Customer Segmentation](/Exp8_6.png)
## FINAL GRAPH(D/O):

![K Means Clustering for Customer Segmentation](/c7.png)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
