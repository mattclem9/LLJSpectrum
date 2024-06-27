# importing libraries
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.cluster import AgglomerativeClustering
import hdbscan

# importing dfs
dfllj_algo = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dflljyear.csv')
dfspec_algo = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dfspecyear.csv')

#? trying to do the sentence transformation/create embeddings, is this correct, are there other things I could do to make it better?
dfllj_algo['Title'] = dfllj_algo['Title'].astype(str)
embeddingsllj = model.encode(dfllj_algo['Title'].tolist())
dfllj_algo['Embeddings'] = list(embeddingsllj)

dfspec_algo['Title'] = dfspec_algo['Title'].astype(str)
embeddingsspec = model.encode(dfspec_algo['Title'].tolist())
dfspec_algo['Embeddings'] = list(embeddingsspec)

#? Creating clusters using AgglomerativeClustering, is this correct? 
num_clusters = 60
agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
dfllj_algo['Clusters'] = agg_clustering.fit_predict(embeddingsllj)

#! Trying to find an easy way to visulize which titles get clustered together 
def visualize_clusters_llj(dfllj_algo, cluster_column='Clusters'):
    unique_clusters = sorted(dfllj_algo[cluster_column].unique())  
    for cluster in unique_clusters:
        cluster_data = dfllj_algo[dfllj_algo[cluster_column] == cluster]['Title'].tolist()
        print(f"Cluster {cluster} Titles: {cluster_data}")

visualize_clusters_llj(dfllj_algo)

# same process but with spectrum 
num_clusters = 60
agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
dfspec_algo['Clusters'] = agg_clustering.fit_predict(embeddingsspec)

# spectrum version of the visualiztion 
def visualize_clusters_spec(dfspec_algo, cluster_column='Clusters'):
    unique_clusters = sorted(dfspec_algo[cluster_column].unique())  
    for cluster in unique_clusters:
        cluster_data = dfspec_algo[dfspec_algo[cluster_column] == cluster]['Title'].tolist()
        print(f"Cluster {cluster} Titles: {cluster_data}")

visualize_clusters_spec(dfspec_algo)
