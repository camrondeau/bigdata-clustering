# load packages to be able to install required packages
import subprocess
import sys

# install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])  
install('pandas')
install('numpy')
install('scikit-learn')
#install('warnings')

# import packages
import pandas as pd
import numpy as np
#import sklearn
#import sklearn.cluster
#import sklearn.metrics.cluster
from sklearn.cluster import KMeans
#from sklearn.metrics.cluster import adjusted_rand_score
#from warnings import simplefilter
#simplefilter(action='ignore', category=FutureWarning)
        
# store file locations as variables
data_file = "/data/p_dsi/big_data_scaling_sp23/project/Clustering/data.csv"
label_file = "/data/p_dsi/big_data_scaling_sp23/project/Clustering/labels.csv"

# read in data file and labels
df = pd.read_csv(data_file)
true_labels = pd.read_csv(label_file, header = None, names = ['contigname', 'true_cluster']) 

# drop label name from data
label_names = df['contigname']
df1 = df.drop('contigname', axis=1)

# set parameters for running BFR algorithm
n_dataload = 50000
k = 745

# randomly sample first subset of data for clustering
df_rand = df1.sample(n=n_dataload, random_state=5)

# remove randmly sampled data from dataset to continue on with clustering
df_rem = pd.concat([df_rand, df1]).drop_duplicates(keep=False)

# convert to numpy and run kmeans
np_rand = df_rand.to_numpy()
kmeans = KMeans(n_clusters=k, random_state=17).fit(np_rand)

# add cluster label to clustered data
df_rand['cluster'] = kmeans.labels_

# group by cluster to store summary statistics for mean and SD
cluster_means = df_rand.groupby('cluster').mean()
cluster_sds = df_rand.groupby('cluster').std()

# add count to keep track of how many point are in each cluster
cluster_means['count'] = df_rand.groupby('cluster').size()
cluster_sds['count'] = df_rand.groupby('cluster').size()

# create an empty data frame to put the points that do not go into a cluster
retained = df_rand.iloc[:0,:].copy().drop('cluster', axis = 1)

# resample another subset of data, remove from dataset
df_rand = df_rem.sample(n=n_dataload, random_state=1)
df_rem = pd.concat([df_rand, df_rem]).drop_duplicates(keep=False) 

# loop through each point in the sample to find out if they should go in a cluster or the retained set
for i in range(0, len(df_rand)):
    # calculate distance of each point from each cluster centroid and find the min value
    distance_list = list(np.sqrt(np.sum(((df_rand.iloc[i,:].values-cluster_means.iloc[:,:-1].values)/cluster_sds.iloc[:, :-1].values)**2, axis =1)))
    min_distance = min(distance_list)
    min_cluster = distance_list.index(min_distance)

    # check if min distance for given row is within 1.5 SD's (sqrt(32)) of closest cluster
    # SD here is just the square root of the number of dimensions
    if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
        # if it is, update cluster mean 
        old_means = cluster_means.iloc[min_cluster,:-1].values
        cluster_means.iloc[min_cluster, :-1] = ((cluster_means.iloc[min_cluster, :-1].values * cluster_means.iloc[min_cluster, cluster_means.shape[1]-1])+df_rand.iloc[i, :].values)/(cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] + 1)
        cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] += 1

        # if it is added to the cluster, update cluster sd 
        n = cluster_sds.iloc[min_cluster, cluster_sds.shape[1]-1]
        old_variance = cluster_sds.iloc[min_cluster, :-1]**2
        new_mean = cluster_means.iloc[min_cluster, :-1]
        new_variance = (((n-1)/n)*old_variance) + ((1/n)*(df_rand.iloc[i, :] - new_mean)*(df_rand.iloc[i, :] - old_means))
        cluster_sds.iloc[min_cluster, :-1] = np.sqrt(new_variance)
        cluster_sds.iloc[min_cluster, cluster_means.shape[1]-1] += 1

    # if it is not, add it to the retained set
    else:
        retained = pd.concat([retained, pd.DataFrame(df_rand.iloc[i]).transpose()])

# cluster the retained set 
k = int(retained.shape[0]*.5)
retained_np = retained.to_numpy()
kmeans = KMeans(n_clusters=k, random_state=10).fit(retained_np)

# save new retained clusters
retained['cluster'] = kmeans.labels_
new_retained_clusters = pd.DataFrame(retained.groupby('cluster').size(), columns=['count_points'])
new_retained_clusters.reset_index(inplace=True)

# check if any cluster has more than 5 points, creating a new compression cluster
compression_dataframe = retained[retained['cluster'].isin(new_retained_clusters[new_retained_clusters['count_points'] > 5]['cluster'])]

# get summary stats of mean and SD for comp clusters, as well as count
comp_cluster_means = compression_dataframe.groupby('cluster').mean()
comp_cluster_sds = compression_dataframe.groupby('cluster').std()
comp_cluster_means['count'] = compression_dataframe.groupby('cluster').size()
comp_cluster_sds['count'] = compression_dataframe.groupby('cluster').size()

# put all points that do not form a compression cluster back into retained
retained = pd.concat([retained, compression_dataframe]).drop_duplicates(keep=False)
retained = retained.drop('cluster', axis = 1)    

# create empty dataframes to store mean and SD for comp clusters that merge with original clusters
remove_comp_means = comp_cluster_means.iloc[:0,:].copy()
remove_comp_sds = comp_cluster_sds.iloc[:0,:].copy()

# loop through each comp cluster to see if they are close and can be merged with original cluster
for i in range(0, len(comp_cluster_means)):
    # calculate distance with each cluster and find min
    distance_list = list(np.sqrt(np.sum(((comp_cluster_means.iloc[i,:-1].values-cluster_means.iloc[:,:-1].values)/cluster_sds.iloc[:, :-1].values)**2, axis =1)))
    min_distance = min(distance_list)
    min_cluster = distance_list.index(min_distance)
    
    # check if min distance for given comp cluster is within 1.5 SD's (sqrt(32)) of closest cluster
    # SD here is just the square root of the number of dimensions
    if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
        # if it is, update cluster mean and SD
        cluster_means.iloc[min_cluster, :-1] = ((cluster_means.iloc[min_cluster, :-1].values * cluster_means.iloc[min_cluster, cluster_means.shape[1]-1])+(comp_cluster_means.iloc[i, :-1].values * comp_cluster_means.iloc[i, comp_cluster_means.shape[1]-1]))/(cluster_means.iloc[min_cluster, cluster_means.shape[1]-1]+comp_cluster_means.iloc[i, comp_cluster_means.shape[1]-1])
        cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] += comp_cluster_means.iloc[i, comp_cluster_means.shape[1]-1]

        cluster_sds.iloc[min_cluster, :-1] = np.sqrt(((cluster_sds.iloc[min_cluster, :-1].values)**2/cluster_sds.iloc[min_cluster, cluster_sds.shape[1]-1])+((comp_cluster_sds.iloc[i, :-1].values)**2/comp_cluster_sds.iloc[i, comp_cluster_sds.shape[1]-1]))
        cluster_sds.iloc[min_cluster, cluster_means.shape[1]-1] += comp_cluster_sds.iloc[i, comp_cluster_sds.shape[1]-1]

        # remove comp clusters that get merged with original clusters
        remove_comp_means = pd.concat([remove_comp_means, pd.DataFrame(comp_cluster_means.iloc[i]).transpose()])
        remove_comp_sds = pd.concat([remove_comp_sds, pd.DataFrame(comp_cluster_sds.iloc[i]).transpose()])

# create new dataframe of comp cluster mean and SDs for all comp clusters that havent been merged
comp_cluster_means = pd.concat([comp_cluster_means, remove_comp_means]).drop_duplicates(keep=False)
comp_cluster_sds = pd.concat([comp_cluster_sds, remove_comp_sds]).drop_duplicates(keep=False)

# loop through remaining dataset, n_dataload rows at a time
while len(df_rem) > n_dataload:
  # randomly sample data and remove from dataset
  df_rand = df_rem.sample(n=n_dataload, random_state=1)
  df_rem = pd.concat([df_rand, df_rem]).drop_duplicates(keep=False) 

  # loop through each point in the sample to find out if they should go in a cluster or the retained set 
  for i in range(0, len(df_rand)):
      # calculate distance of each point from each cluster centroid and find the min value
      distance_list = list(np.sqrt(np.sum(((df_rand.iloc[i,:].values-cluster_means.iloc[:,:-1].values)/cluster_sds.iloc[:, :-1].values)**2, axis =1)))
      min_distance = min(distance_list)
      min_cluster = distance_list.index(min_distance)

      # check if min distance for given row is within 1.5 SD's (sqrt(32)) of closest cluster
      # SD here is just the square root of the number of dimensions
      if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
          # if it is, update cluster mean 
          old_means = cluster_means.iloc[min_cluster,:-1].values
          cluster_means.iloc[min_cluster, :-1] = ((cluster_means.iloc[min_cluster, :-1].values * cluster_means.iloc[min_cluster, cluster_means.shape[1]-1])+df_rand.iloc[i, :].values)/(cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] + 1)
          cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] += 1

          # if it is added to the cluster, update cluster sd 
          n = cluster_sds.iloc[min_cluster, cluster_sds.shape[1]-1]
          old_variance = cluster_sds.iloc[min_cluster, :-1]**2
          new_mean = cluster_means.iloc[min_cluster, :-1]
          new_variance = (((n-1)/n)*old_variance) + ((1/n)*(df_rand.iloc[i, :] - new_mean)*(df_rand.iloc[i, :] - old_means))
          cluster_sds.iloc[min_cluster, :-1] = np.sqrt(new_variance)
          cluster_sds.iloc[min_cluster, cluster_means.shape[1]-1] += 1

      # if it is not, check if it can be merged to any comp cluster
      else:
          if len(comp_cluster_means) > 0:
              # calculate distance of each point from each comp cluster centroid and find the min value
              distance_list = list(np.sqrt(np.sum(((df_rand.iloc[i,:].values-comp_cluster_means.iloc[:,:-1].values)/comp_cluster_sds.iloc[:, :-1].values)**2, axis =1)))
              min_distance = min(distance_list)
              min_cluster = distance_list.index(min_distance)

              # check if min distance for given row is within 2 SD's (sqrt(32)) of closest cluster
              # SD here is just the square root of the number of dimensions
              if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
                  # if it is, update cluster mean 
                  old_means = comp_cluster_means.iloc[min_cluster,:-1].values
                  comp_cluster_means.iloc[min_cluster, :-1] = ((comp_cluster_means.iloc[min_cluster, :-1].values * comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1])+df_rand.iloc[i, :].values)/(comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1] + 1)
                  comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1] += 1

                  # if it is added to the cluster, update cluster sd 
                  n = comp_cluster_sds.iloc[min_cluster, comp_cluster_sds.shape[1]-1]
                  old_variance = comp_cluster_sds.iloc[min_cluster, :-1]**2
                  new_mean = comp_cluster_means.iloc[min_cluster, :-1]
                  new_variance = (((n-1)/n)*old_variance) + ((1/n)*(df_rand.iloc[i, :] - new_mean)*(df_rand.iloc[i, :] - old_means))
                  comp_cluster_sds.iloc[min_cluster, :-1] = np.sqrt(new_variance)
                  comp_cluster_sds.iloc[min_cluster, comp_cluster_means.shape[1]-1] += 1
              # if it is not close to either cluster, add back to retained
              else:
                  retained = pd.concat([retained, pd.DataFrame(df_rand.iloc[i]).transpose()])
          else:
              retained = pd.concat([retained, pd.DataFrame(df_rand.iloc[i]).transpose()])

  # cluster the retained set 
  k = int(retained.shape[0]*.5)
  retained_np = retained.to_numpy()
  kmeans = KMeans(n_clusters=k, random_state=10).fit(retained_np)

  # save new retained clusters
  retained['cluster'] = kmeans.labels_
  new_retained_clusters = pd.DataFrame(retained.groupby('cluster').size(), columns=['count_points'])
  new_retained_clusters.reset_index(inplace=True)
  
  # check if any cluster has more than 5 points, creating a new compression cluster
  new_clusters = retained[retained['cluster'].isin(new_retained_clusters[new_retained_clusters['count_points'] > 5]['cluster'])]

  # get summary stats of mean and SD for comp clusters, as well as count
  new_cluster_means = new_clusters.groupby('cluster').mean()
  new_cluster_sds = new_clusters.groupby('cluster').std()
  new_cluster_means['count'] = new_clusters.groupby('cluster').size()
  new_cluster_sds['count'] = new_clusters.groupby('cluster').size()

  # put all points that do not form a compression cluster back into retained
  retained = pd.concat([retained, new_clusters]).drop_duplicates(keep=False)
  retained = retained.drop('cluster', axis = 1)    

  # create empty dataframes to store mean and SD for comp clusters that merge with original clusters
  remove_comp_means = comp_cluster_means.iloc[:0,:].copy()
  remove_comp_sds = comp_cluster_sds.iloc[:0,:].copy()

  # loop through each of the new clusters to check if they can be merged with original cluster or comp cluster
  for i in range(0, len(new_cluster_means)):
      # calculate distance with each cluster and find min
      distance_list = list(np.sqrt(np.sum(((new_cluster_means.iloc[i,:-1].values-cluster_means.iloc[:,:-1].values)/cluster_sds.iloc[:, :-1].values)**2, axis =1)))
      min_distance = min(distance_list)
      min_cluster = distance_list.index(min_distance)
      
      # check if min distance for given retained cluster is within 1.5 SD's (sqrt(32)) of closest cluster
      # SD here is just the square root of the number of dimensions
      if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
          # if it is, update mean and sd
          cluster_means.iloc[min_cluster, :-1] = ((cluster_means.iloc[min_cluster, :-1].values * cluster_means.iloc[min_cluster, cluster_means.shape[1]-1])+(new_cluster_means.iloc[i, :-1].values * new_cluster_means.iloc[i, new_cluster_means.shape[1]-1]))/(cluster_means.iloc[min_cluster, cluster_means.shape[1]-1]+new_cluster_means.iloc[i, new_cluster_means.shape[1]-1])
          cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] += new_cluster_means.iloc[i, new_cluster_means.shape[1]-1]

          cluster_sds.iloc[min_cluster, :-1] = np.sqrt(((cluster_sds.iloc[min_cluster, :-1].values)**2/cluster_sds.iloc[min_cluster, cluster_sds.shape[1]-1])+((new_cluster_sds.iloc[i, :-1].values)**2/new_cluster_sds.iloc[i, new_cluster_sds.shape[1]-1]))
          cluster_sds.iloc[min_cluster, cluster_means.shape[1]-1] += new_cluster_sds.iloc[i, new_cluster_sds.shape[1]-1]

          # remove comp clusters that get merged with original clusters
          remove_comp_means = pd.concat([remove_comp_means, pd.DataFrame(new_cluster_means.iloc[i]).transpose()])
          remove_comp_sds = pd.concat([remove_comp_sds, pd.DataFrame(new_cluster_sds.iloc[i]).transpose()])
      
      else:
          # if it is not, check if it is close to any existing comp clusters, only if there are comp clusters
          if len(comp_cluster_means) > 0:
              # calculate distance with each cluster and find min
              distance_list = list(np.sqrt(np.sum(((new_cluster_means.iloc[i,:-1].values-comp_cluster_means.iloc[:,:-1].values)/comp_cluster_sds.iloc[:, :-1].values)**2, axis =1)))
              min_distance = min(distance_list)
              min_cluster = distance_list.index(min_distance)

              # check if min distance for given retained cluster is within 1.5 SD's (sqrt(32)) of closest cluster
              # SD here is just the square root of the number of dimensions
              if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
                  # if it is, update mean and SD
                  comp_cluster_means.iloc[min_cluster, :-1] = ((comp_cluster_means.iloc[min_cluster, :-1].values * comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1])+(new_cluster_means.iloc[i, :-1].values * new_cluster_means.iloc[i, new_cluster_means.shape[1]-1]))/(comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1]+new_cluster_means.iloc[i, new_cluster_means.shape[1]-1])
                  comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1] += new_cluster_means.iloc[i, new_cluster_means.shape[1]-1]

                  comp_cluster_sds.iloc[min_cluster, :-1] = np.sqrt(((comp_cluster_sds.iloc[min_cluster, :-1].values)**2/comp_cluster_sds.iloc[min_cluster, comp_cluster_sds.shape[1]-1])+((new_cluster_sds.iloc[i, :-1].values)**2/new_cluster_sds.iloc[i, new_cluster_sds.shape[1]-1]))
                  comp_cluster_sds.iloc[min_cluster, comp_cluster_means.shape[1]-1] += new_cluster_sds.iloc[i, new_cluster_sds.shape[1]-1]

                  # remove comp clusters that get merged with original clusters
                  remove_comp_means = pd.concat([remove_comp_means, pd.DataFrame(new_cluster_means.iloc[i]).transpose()])
                  remove_comp_sds = pd.concat([remove_comp_sds, pd.DataFrame(new_cluster_sds.iloc[i]).transpose()])

  # drop any new clusters that get merged
  new_cluster_means = pd.concat([new_cluster_means, remove_comp_means]).drop_duplicates(keep=False)
  new_cluster_sds = pd.concat([new_cluster_sds, remove_comp_sds]).drop_duplicates(keep=False)
  
  # add those to comp clusters as they become their own comp cluster
  comp_cluster_means = pd.concat([comp_cluster_means, new_cluster_means], ignore_index=True)
  comp_cluster_sds = pd.concat([comp_cluster_sds, new_cluster_sds], ignore_index=True)
    

# rerun everything on remaining data
df_rand = df_rem

# loop through each point in the sample to find out if they should go in a cluster or the retained set 
for i in range(0, len(df_rand)):
    # calculate distance with each cluster and find min
    distance_list = list(np.sqrt(np.sum(((df_rand.iloc[i,:].values-cluster_means.iloc[:,:-1].values)/cluster_sds.iloc[:, :-1].values)**2, axis =1)))
    min_distance = min(distance_list)
    min_cluster = distance_list.index(min_distance)

    # check if min distance for given row is within 1.5 SD's (sqrt(32)) of closest cluster
    # SD here is just the square root of the number of dimensions
    if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
        # if it is, update cluster mean 
        old_means = cluster_means.iloc[min_cluster,:-1].values
        cluster_means.iloc[min_cluster, :-1] = ((cluster_means.iloc[min_cluster, :-1].values * cluster_means.iloc[min_cluster, cluster_means.shape[1]-1])+df_rand.iloc[i, :].values)/(cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] + 1)
        cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] += 1

        # if it is added to the cluster, update cluster sd 
        n = cluster_sds.iloc[min_cluster, cluster_sds.shape[1]-1]
        old_variance = cluster_sds.iloc[min_cluster, :-1]**2
        new_mean = cluster_means.iloc[min_cluster, :-1]
        new_variance = (((n-1)/n)*old_variance) + ((1/n)*(df_rand.iloc[i, :] - new_mean)*(df_rand.iloc[i, :] - old_means))
        cluster_sds.iloc[min_cluster, :-1] = np.sqrt(new_variance)
        cluster_sds.iloc[min_cluster, cluster_means.shape[1]-1] += 1

    # if it is not, check if it is close to comp cluster
    else:
        # loop through comp clusters
        if len(comp_cluster_means) > 0:
            # calculate distance to each comp cluster and save min
            distance_list = list(np.sqrt(np.sum(((df_rand.iloc[i,:].values-comp_cluster_means.iloc[:,:-1].values)/comp_cluster_sds.iloc[:, :-1].values)**2, axis =1)))
            min_distance = min(distance_list)
            min_cluster = distance_list.index(min_distance)

            # check if min distance for given row is within 1.5 SD's (sqrt(32)) of closest cluster
            # SD here is just the square root of the number of dimensions
            if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
                # if it is, update cluster mean 
                old_means = comp_cluster_means.iloc[min_cluster,:-1].values
                comp_cluster_means.iloc[min_cluster, :-1] = ((comp_cluster_means.iloc[min_cluster, :-1].values * comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1])+df_rand.iloc[i, :].values)/(comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1] + 1)
                comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1] += 1

                # if it is added to the cluster, update cluster sd 
                n = comp_cluster_sds.iloc[min_cluster, comp_cluster_sds.shape[1]-1]
                old_variance = comp_cluster_sds.iloc[min_cluster, :-1]**2
                new_mean = comp_cluster_means.iloc[min_cluster, :-1]
                new_variance = (((n-1)/n)*old_variance) + ((1/n)*(df_rand.iloc[i, :] - new_mean)*(df_rand.iloc[i, :] - old_means))
                comp_cluster_sds.iloc[min_cluster, :-1] = np.sqrt(new_variance)
                comp_cluster_sds.iloc[min_cluster, comp_cluster_means.shape[1]-1] += 1

            # if it is not, add to retained set
            else:
                retained = pd.concat([retained, pd.DataFrame(df_rand.iloc[i]).transpose()])
        # if it is not, add to retained set
        else:
            retained = pd.concat([retained, pd.DataFrame(df_rand.iloc[i]).transpose()])
      
# cluster the retained set 
k = int(retained.shape[0]*.5)
retained_np = retained.to_numpy()
kmeans = KMeans(n_clusters=k, random_state=10).fit(retained_np)

# save new retained clusters
retained['cluster'] = kmeans.labels_
new_retained_clusters = pd.DataFrame(retained.groupby('cluster').size(), columns=['count_points'])
new_retained_clusters.reset_index(inplace=True)

# check if any cluster has more than 5 points, creating a new compression cluster
new_clusters = retained[retained['cluster'].isin(new_retained_clusters[new_retained_clusters['count_points'] > 5]['cluster'])]

# get cluster mean, SD, and count
new_cluster_means = new_clusters.groupby('cluster').mean()
new_cluster_sds = new_clusters.groupby('cluster').std()
new_cluster_means['count'] = new_clusters.groupby('cluster').size()
new_cluster_sds['count'] = new_clusters.groupby('cluster').size()

# put all points that do not form a compression cluster back into retained
retained = pd.concat([retained, new_clusters]).drop_duplicates(keep=False)
retained = retained.drop('cluster', axis = 1)    

# create empty dataframes to store mean and SD for comp clusters that merge with original clusters
remove_comp_means = comp_cluster_means.iloc[:0,:].copy()
remove_comp_sds = comp_cluster_sds.iloc[:0,:].copy()
     
# loop through new clusters to see if they can be merged with original cluster or comp cluster
for i in range(0, len(new_cluster_means)):
    # calculate distance and save min
    distance_list = list(np.sqrt(np.sum(((new_cluster_means.iloc[i,:-1].values-cluster_means.iloc[:,:-1].values)/cluster_sds.iloc[:, :-1].values)**2, axis =1)))
    min_distance = min(distance_list)
    min_cluster = distance_list.index(min_distance)
    
    # check if min distance for given row is within 1.5 SD's (sqrt(32)) of closest cluster
    # SD here is just the square root of the number of dimensions
    if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
        # if it is, save mean and SD
        cluster_means.iloc[min_cluster, :-1] = ((cluster_means.iloc[min_cluster, :-1].values * cluster_means.iloc[min_cluster, cluster_means.shape[1]-1])+(new_cluster_means.iloc[i, :-1].values * new_cluster_means.iloc[i, new_cluster_means.shape[1]-1]))/(cluster_means.iloc[min_cluster, cluster_means.shape[1]-1]+new_cluster_means.iloc[i, new_cluster_means.shape[1]-1])
        cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] += new_cluster_means.iloc[i, new_cluster_means.shape[1]-1]

        cluster_sds.iloc[min_cluster, :-1] = np.sqrt(((cluster_sds.iloc[min_cluster, :-1].values)**2/cluster_sds.iloc[min_cluster, cluster_sds.shape[1]-1])+((new_cluster_sds.iloc[i, :-1].values)**2/new_cluster_sds.iloc[i, new_cluster_sds.shape[1]-1]))
        cluster_sds.iloc[min_cluster, cluster_means.shape[1]-1] += new_cluster_sds.iloc[i, new_cluster_sds.shape[1]-1]

        # remove comp clusters that get merged
        remove_comp_means = pd.concat([remove_comp_means, pd.DataFrame(new_cluster_means.iloc[i]).transpose()])
        remove_comp_sds = pd.concat([remove_comp_sds, pd.DataFrame(new_cluster_sds.iloc[i]).transpose()])
    
    else:
        # if it is not, check if it is close to any existing comp clusters, only if there are comp clusters
        if len(comp_cluster_means) > 0:
            # calculate distance save min
            distance_list = list(np.sqrt(np.sum(((new_cluster_means.iloc[i,:-1].values-comp_cluster_means.iloc[:,:-1].values)/comp_cluster_sds.iloc[:, :-1].values)**2, axis =1)))
            min_distance = min(distance_list)
            min_cluster = distance_list.index(min_distance)

            # check if min distance for given row is within 1.5 SD's (sqrt(32)) of closest cluster
            # SD here is just the square root of the number of dimensions
            if min_distance < (np.sqrt(df_rand.shape[1])*1.5):
                # if it is update mean and SD
                comp_cluster_means.iloc[min_cluster, :-1] = ((comp_cluster_means.iloc[min_cluster, :-1].values * comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1])+(new_cluster_means.iloc[i, :-1].values * new_cluster_means.iloc[i, new_cluster_means.shape[1]-1]))/(comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1]+new_cluster_means.iloc[i, new_cluster_means.shape[1]-1])
                comp_cluster_means.iloc[min_cluster, comp_cluster_means.shape[1]-1] += new_cluster_means.iloc[i, new_cluster_means.shape[1]-1]

                comp_cluster_sds.iloc[min_cluster, :-1] = np.sqrt(((comp_cluster_sds.iloc[min_cluster, :-1].values)**2/comp_cluster_sds.iloc[min_cluster, comp_cluster_sds.shape[1]-1])+((new_cluster_sds.iloc[i, :-1].values)**2/new_cluster_sds.iloc[i, new_cluster_sds.shape[1]-1]))
                comp_cluster_sds.iloc[min_cluster, comp_cluster_means.shape[1]-1] += new_cluster_sds.iloc[i, new_cluster_sds.shape[1]-1]

                remove_comp_means = pd.concat([remove_comp_means, pd.DataFrame(new_cluster_means.iloc[i]).transpose()])
                remove_comp_sds = pd.concat([remove_comp_sds, pd.DataFrame(new_cluster_sds.iloc[i]).transpose()])

new_cluster_means = pd.concat([new_cluster_means, remove_comp_means]).drop_duplicates(keep=False)
new_cluster_sds = pd.concat([new_cluster_sds, remove_comp_sds]).drop_duplicates(keep=False)

comp_cluster_means = pd.concat([comp_cluster_means, new_cluster_means], ignore_index=True)
comp_cluster_sds = pd.concat([comp_cluster_sds, new_cluster_sds], ignore_index=True)

# assign final retained to nearest cluster
for i in range(0, len(retained)):
    # each point will have a distance from all cluster centroids
    distance_list = list(np.sqrt(np.sum(((retained.iloc[i,:].values-cluster_means.iloc[:,:-1].values)/cluster_sds.iloc[:, :-1].values)**2, axis =1)))
    min_distance = min(distance_list)
    min_cluster = distance_list.index(min_distance)

    # if it is, update cluster mean 
    old_means = cluster_means.iloc[min_cluster,:-1].values
    cluster_means.iloc[min_cluster, :-1] = ((cluster_means.iloc[min_cluster, :-1].values * cluster_means.iloc[min_cluster, cluster_means.shape[1]-1])+retained.iloc[i, :].values)/(cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] + 1)
    cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] += 1

    # if it is added to the cluster, update cluster sd 
    n = cluster_sds.iloc[min_cluster, cluster_sds.shape[1]-1]
    old_variance = cluster_sds.iloc[min_cluster, :-1]**2
    new_mean = cluster_means.iloc[min_cluster, :-1]
    new_variance = (((n-1)/n)*old_variance) + ((1/n)*(retained.iloc[i, :] - new_mean)*(retained.iloc[i, :] - old_means))
    cluster_sds.iloc[min_cluster, :-1] = np.sqrt(new_variance)
    cluster_sds.iloc[min_cluster, cluster_means.shape[1]-1] += 1
    
# assign final comp clusters to nearest cluster
for i in range(0, len(comp_cluster_means)):
    distance_list = list(np.sqrt(np.sum(((comp_cluster_means.iloc[i,:-1].values-cluster_means.iloc[:,:-1].values)/cluster_sds.iloc[:, :-1].values)**2, axis =1)))
    min_distance = min(distance_list)
    min_cluster = distance_list.index(min_distance)

    cluster_means.iloc[min_cluster, :-1] = ((cluster_means.iloc[min_cluster, :-1].values * cluster_means.iloc[min_cluster, cluster_means.shape[1]-1])+(comp_cluster_means.iloc[i, :-1].values * comp_cluster_means.iloc[i, comp_cluster_means.shape[1]-1]))/(cluster_means.iloc[min_cluster, cluster_means.shape[1]-1]+comp_cluster_means.iloc[i, comp_cluster_means.shape[1]-1])
    cluster_means.iloc[min_cluster, cluster_means.shape[1]-1] += comp_cluster_means.iloc[i, comp_cluster_means.shape[1]-1]

    cluster_sds.iloc[min_cluster, :-1] = np.sqrt(((cluster_sds.iloc[min_cluster, :-1].values)**2/cluster_sds.iloc[min_cluster, cluster_sds.shape[1]-1])+((comp_cluster_sds.iloc[i, :-1].values)**2/comp_cluster_sds.iloc[i, comp_cluster_sds.shape[1]-1]))
    cluster_sds.iloc[min_cluster, cluster_means.shape[1]-1] += comp_cluster_sds.iloc[i, comp_cluster_sds.shape[1]-1]
    
# print results
print('Total clustered points:', cluster_means['count'].sum())
print('Total number of clusters:', len(cluster_means))
print('First 5 clusters:')
print(cluster_means.head(5))