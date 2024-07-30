import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

dfllj_algo = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dfflljcluster1.csv')
dfllj_algo['Title'] = dfllj_algo['Title'].astype(str)+'.'

model = SentenceTransformer('all-MiniLM-L6-v2')
dfllj_algo['Title'] = dfllj_algo['Title'].astype(str)
embeddingsllj = model.encode(dfllj_algo['Title'].tolist())
dfllj_algo['Embeddings'] = list(embeddingsllj)


def save_clusters_to_file(df, cluster_column, filename):
    unique_clusters = sorted(df[cluster_column].unique())
    with open(filename, 'w') as file:
        for cluster in unique_clusters:
            cluster_data = df[df[cluster_column] == cluster]['Title'].tolist()
            file.write(f"Cluster {cluster}:\n")
            for title in cluster_data:
                file.write(f"    {title}\n")
            file.write("\n")


distance_thresholds = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5]  

for distance_threshold in distance_thresholds:
    agg_clustering_llj = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        affinity='cosine', 
        linkage='average',
        compute_full_tree=True
    )
    dfllj_algo['Clusters'] = agg_clustering_llj.fit_predict(embeddingsllj)
    
    file_name = f"AggClusteringLLJPeriod_threshold_{distance_threshold}.txt"
    save_clusters_to_file(dfllj_algo, 'Clusters', file_name)
    print(f"Clusters have been saved to {file_name}")



