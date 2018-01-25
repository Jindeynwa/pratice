import pandas as pd
import quandl
import math ,datetime
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.cluster import KMeans
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

clf = KMeans(n_clusters=2)
clf.fit(X)



correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    print ("predict me" +str(predict_me))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(float(correct*100/len(X)))

a= np.array([6,5,4,5]);
a = a.reshape(-1,len(a))
print(a)




