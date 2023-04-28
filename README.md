# Large Scale Clustering
**Sydney Simmons**
**Cameron Rondeau**

## Project Introduction
* Clustering is an effective algorithm to put points into groups so that points within each group are similar 
* Once the points are in clusters, you can gain insights into the trends or similarities within groups, which you would not be able to see otherwise, especially in large dimensional data
* However, clustering is a high memory operation because the data for each point and the cluster that it is assigned to is saved 
* The BFR algorithm reduces the memory of clustering, even with large dimensional data, by saving statistics for each cluster instead of the points in each cluster
* During this project, we implemented both regular clustering and BFR algorithm clustering and compared memory usage 

## Milestone 1
* Utilize K-Means clustering algorithm on largescale dataset using ACCRE
  * Created 745 clusters
  * Adjusted RAND Index Score: 0.262
  * Peak Memory Usage: 10.4GB

## Milestone 2
* Implement Bradley, Fayyad and Reina (BFR) Algorithm
  * Step 1: Input subset of points into KMeans
  * Step 2: Randomly sample a new set of points and add them to existing clusters, if close
  * Step 3: If a point is added to an existing cluster, update the mean and standard deviation and discard point
  * Step 4: For the points not close enough to a cluster, cluster them using KMeans and add clusters to a compression set
  * Step 5: Check if compression set clusters can be put into existing 745 clusters or other compression set clusters
  * Step 6: Loop through remaining points 50,000 at a time and repeat until there are no more points remaining
  * Step 7: Recluster the retained set after every 50,000 point sample
  * Step 8: Add compression set clusters and remaining retained points to closest existing cluster

### Results
* BFR Algorithm had a peak memory usage of 2.2GB, which is an ~80% reduction to standard K-Means

## Conclusion
* Using the BFR clustering algorithm requires much less storage than the normal K-means clustering algorithm
* By keeping summary statistics of clusters, similar clustering results can be obtained by only looking at small subsets of the data at a time
* Vectorizing data results in much faster and more efficient clustering

