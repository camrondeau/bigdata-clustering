import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install('pandas')
install('numpy')
install('scikit-learn')

import pandas as pd
import numpy as np
#import sklearn
#import sklearn.cluster
#import sklearn.metrics.cluster
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

# store file locations as variables
data_file = "/data/p_dsi/big_data_scaling_sp23/project/Clustering/data.csv"
label_file = "/data/p_dsi/big_data_scaling_sp23/project/Clustering/labels.csv"

# read in files
df = pd.read_csv(data_file)
true_labels = pd.read_csv(label_file, header = None, names = ['contigname', 'true_cluster']) 

# store label names
label_names = df['contigname']

# drop label names from df and store as numpy array
np_df = df.drop('contigname', axis=1).to_numpy()

# run k-means
kmeans = KMeans(n_clusters=745, random_state=2023).fit(np_df)

# create dataframe with label names and predicted clusters
df_k = pd.DataFrame({'contigname': label_names,'predicted_cluster': kmeans.labels_,})

# merge above dataframe with true clusters and drop NA's
cluster_df = df_k.merge(true_labels, on='contigname', how='left').dropna()

# get predicted clusters and true clusters
p_cluster = list(cluster_df['predicted_cluster'])
t_cluster = list(cluster_df['true_cluster'])

# calculate adjusted rand score
ars = adjusted_rand_score(t_cluster, p_cluster)

print('ARI: ', str(ars))