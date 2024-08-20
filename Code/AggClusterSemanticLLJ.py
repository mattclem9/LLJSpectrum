import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Load your DataFrame
dfllj_algo = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dflljclusterNomask.csv')
dfllj_algo['Title'] = dfllj_algo['Title'].astype(str)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the titles
embeddingsllj = model.encode(dfllj_algo['Title'].tolist(), show_progress_bar=True)
dfllj_algo['Embeddings'] = list(embeddingsllj)

# Function to save clusters to a file
def save_clusters_to_file(df, cluster_column, filename):
    unique_clusters = sorted(df[cluster_column].unique())
    with open(filename, 'w') as file:
        for cluster in unique_clusters:
            cluster_data = df[df[cluster_column] == cluster]['Title'].tolist()
            file.write(f"Cluster {cluster}:\n")
            for title in cluster_data:
                file.write(f"    {title}\n")
            file.write("\n")

# List of cluster sizes to try
n_clusters_list = [10, 20, 30, 40, 50, 60, 70]

# Loop through the different cluster sizes
for n_clusters in n_clusters_list:
    agg_clustering_llj = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='cosine', 
        linkage='average',
    )
    
    
    dfllj_algo['Clusters'] = agg_clustering_llj.fit_predict(embeddingsllj)
    
   
    file_name = f"AggClusteringLLJExplode_{n_clusters}.txt"
    save_clusters_to_file(dfllj_algo, 'Clusters', file_name)
    print(f"Clusters have been saved to {file_name}")
    
 
    dfllj_algo['Cluster'] = 'Cluster ' + dfllj_algo['Clusters'].astype(str)


