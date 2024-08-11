import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np


dfllj = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dflljcluster1.csv')
dfllj['normalized_title'] = dfllj['Title'].astype(str) + '.'


model = SentenceTransformer('all-MiniLM-L6-v2')
titles = dfllj['normalized_title'].tolist()
embeddings = model.encode(titles, show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, metric='euclidean', cluster_selection_method='leaf')
cluster_labels = clusterer.fit_predict(embeddings)

print("Cluster labels:", np.unique(cluster_labels, return_counts=True))


dfllj['cluster'] = cluster_labels

clusters = {}
for cluster in np.unique(cluster_labels):
    clusters[cluster] = dfllj[dfllj['cluster'] == cluster]['Title'].tolist()

def save_clusters_to_file(clusters, filename):
    with open(filename, 'w') as file:
        for cluster_id, titles in clusters.items():
            file.write(f"Cluster {cluster_id}:\n")
            for title in titles:
                file.write(f"    {title}\n")
            file.write("\n")

filename = 'HDBScanLLJSelect5minleaf.txt'
save_clusters_to_file(clusters, filename)

print("Clusters have been saved to", filename)


for cluster_id, titles in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for title in titles:
        print(title)
