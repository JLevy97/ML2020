import LoadData

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

np.random.seed(123)

def k_means(n_clust, data_frame, true_labels):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels
    and clustering performance parameters.

    Input:
    n_clust - number of clusters (k value)
    data_frame - dataset we want to cluster
    true_labels - original labels

    Output:
    1 - crosstab of cluster and actual labels
    2 - performance table
    """
    k_means = KMeans(n_clusters=n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (k_means.inertia_,
             homogeneity_score(true_labels, y_clust),
             completeness_score(true_labels, y_clust),
             v_measure_score(true_labels, y_clust),
             adjusted_rand_score(true_labels, y_clust),
             adjusted_mutual_info_score(true_labels, y_clust),
             silhouette_score(data_frame, y_clust, metric='euclidean')))

#drops unneeded features from the dataframe and stores them in another set
def featureEngineering(x):

    #drop score from dataset
    Data = x.drop(['Score'], axis=1)

    #store ranks and nations
    Zed = Data[['Overall rank','Country or region']]
    Data = Data.drop(['Country or region', 'Overall rank'], axis=1)

    #print(Data)
    #print(Zed)

    return Data,Zed

#given a set of scores, splits the data into brackets to justify groups of similar hapinness scores
def bracketScores(y, numbrackets):

    buckets = np.zeros(y.shape)

    mx = np.max(y)
    mn = np.min(y)

    length_of_range = (mx - mn + 1) / numbrackets

    rangeArray = np.zeros([numbrackets,2])
    start_of_range = mn
    end_of_range = start_of_range + length_of_range
    i=0
    rangeArray[i] = [start_of_range,end_of_range]
    while i<numbrackets-1:
        start_of_range = end_of_range
        end_of_range = start_of_range + length_of_range
        i = i+1
        rangeArray[i] = [start_of_range, end_of_range]

    #print(mn)
    #print(mx)
    #print(rangeArray)

    for i in range(len(y)):
        val = y[i]
        brac = 0
        for j in range(len(rangeArray)):
            if val>= rangeArray[j][0] and val<rangeArray[j][1]:
                brac = j
                break
        buckets[i] = brac

    #print(y)
    #print(buckets)

    return buckets

#plots the inertia graph in order to best determine the number of clusters for K-means
def plotOptimalK(x):

    # check the optimal k value
    ks = range(1, 10)
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(x)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.style.use('bmh')
    plt.plot(ks, inertias, '-o')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.xticks(ks)
    plt.show()

    return


if __name__ == '__main__':
    x2019, y2019 = LoadData.load2019()
    x2019,z2019 = featureEngineering(x2019)

    plotOptimalK(x2019) #results show kinks in k = 2,3

    yBracket2 = bracketScores(y2019, 2)
    yBracket3 = bracketScores(y2019, 3)

    # 2 cluster
    print("\nKmeans with 2 clusters")
    k_means(n_clust=2, data_frame=x2019, true_labels=yBracket2)

    print()
    print("\nKmeans with 3 clusters")
    # 3 clusters
    k_means(n_clust=3, data_frame=x2019, true_labels=yBracket3)

    print()
    print("\nKmeans with 6 clusters")
    # 6 clusters
    k_means(n_clust=6, data_frame=x2019, true_labels=bracketScores(y2019, 6))
