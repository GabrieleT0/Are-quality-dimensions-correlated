import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
import glob

csv_files = glob.glob('../data/inferential_statistics_results/*_Always_UP.csv') 
corr_matrices = []

for file in csv_files:
    df = pd.read_csv(file, index_col=0)
    corr_df = df.loc[:, ~df.columns.str.contains('p-value')]
    corr_df = corr_df.loc[corr_df.columns]
    corr_matrices.append(corr_df.astype(float))

avg_corr_matrix = sum(corr_matrices) / len(corr_matrices)

# Convert correlation to distance matrix
distance_matrix = 1 - avg_corr_matrix
condensed_dist = sch.distance.squareform(distance_matrix.values)

linkage = sch.linkage(condensed_dist, method='average')

num_clusters = 4 # Can change this
cluster_labels = fcluster(linkage, t=num_clusters, criterion='maxclust')

dimension_names = avg_corr_matrix.index.tolist()
cluster_assignments = dict(zip(dimension_names, cluster_labels))

print("Cluster assignments:")
for dim, label in cluster_assignments.items():
    print(f"{dim}: Cluster {label}")

sns.clustermap(avg_corr_matrix, row_linkage=linkage, col_linkage=linkage,
               cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt=".2f", figsize=(12, 10))
plt.title("Hierarchical Clustering of Quality Dimensions")
plt.savefig('../data/charts/hierarchical_clustering')