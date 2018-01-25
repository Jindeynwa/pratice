import pandas as pd
import quandl
import math ,datetime
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.cluster import MeanShift
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# X =np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,1.06],[9,11]])
# plt.scatter(X[:,0],X[:,1],s=150)
#
# clf = KMeans(6)
# clf.fit(X)
#
# centroid = clf.cluster_centers_
# labels = clf.labels_
#
# print ("labels"+str(labels))
# print ("centrid"+str(centroid))
# colors = 10*["g.","r.","c.","b.","k."]
# for i in range(len(X)):
#     plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize = 10)
#
# plt.scatter(centroid[:,0],centroid[:,1],marker = 'X' ,s=150,linewidths=5)
# plt.show()

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df
df = handle_non_numerical_data(df)
df.drop(['sex','boat'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
survival_rates ={}

for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df["survived"]==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print survival_rates



