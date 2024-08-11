import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np


dfllj_algo = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dflljcluster1.csv')
dfllj_algo['Title'] = dfllj_algo['Title']


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean pooling of the last hidden states as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings


dfllj_algo['Title'] = dfllj_algo['Title'].astype(str)
embeddingsllj = get_embeddings(dfllj_algo['Title'].tolist())
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


n_clusters_list = [10, 20, 30, 40, 50, 60, 71]

for n_clusters in n_clusters_list:
    agg_clustering_llj = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='cosine', 
        linkage='average',
    )
    dfllj_algo['Clusters'] = agg_clustering_llj.fit_predict(embeddingsllj)
    
    # Save clusters to a text file
    file_name = f"AggClusteringLLJ_BERT{n_clusters}.txt"
    save_clusters_to_file(dfllj_algo, 'Clusters', file_name)
    print(f"Clusters have been saved to {file_name}")
    
    # Add Cluster column with 'Cluster X' format
    dfllj_algo['Cluster'] = 'Cluster ' + dfllj_algo['Clusters'].astype(str)



