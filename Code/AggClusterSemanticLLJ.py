import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import nltk

dfllj_algo = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dfflljcluster1.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

dfllj_algo['Title'] = dfllj_algo['Title'].astype(str)
embeddingsllj = model.encode(dfllj_algo['Title'].tolist())
dfllj_algo['Embeddings'] = list(embeddingsllj)

num_clusters_llj = 20
agg_clustering_llj = AgglomerativeClustering(n_clusters=num_clusters_llj, affinity = 'cosine', linkage='average')
dfllj_algo['Clusters'] = agg_clustering_llj.fit_predict(embeddingsllj)

def save_clusters_to_file(df, cluster_column, filename):
    unique_clusters = sorted(df[cluster_column].unique())
    with open(filename, 'w') as file:
        for cluster in unique_clusters:
            cluster_data = df[df[cluster_column] == cluster]['Title'].tolist()
            file.write(f"Cluster {cluster}:\n")
            for title in cluster_data:
                file.write(f"    {title}\n")
            file.write("\n")

save_clusters_to_file(dfllj_algo, 'Clusters', 'AggClusteringLLJSelect20clusters.txt')




