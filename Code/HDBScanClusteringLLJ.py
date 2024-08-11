import pandas as pd
import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


dfllj = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dflljcluster1.csv')
dfllj['normalized_title'] = dfllj['Title'].astype(str) + '.'


model = SentenceTransformer('all-MiniLM-L6-v2')

titles = dfllj['normalized_title'].tolist()
embeddings = model.encode(titles, show_progress_bar=True)



clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, metric='euclidean', cluster_selection_method='eom', alpha=1.0)
cluster_labels = clusterer.fit_predict(reduced_embeddings)


dfllj['cluster'] = cluster_labels


cluster_counts = np.unique(cluster_labels, return_counts=True)
print("Cluster counts:", dict(zip(cluster_counts[0], cluster_counts[1])))


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

filename = 'HDBScanLLJ_euclidean.txt'
save_clusters_to_file(clusters, filename)

print(f"Clusters have been saved to {filename}")
