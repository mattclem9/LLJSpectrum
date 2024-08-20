import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Load your DataFrame
dfspec_algo = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dspeccluster1.csv')
dfspec_algo['Title'] = dfspec_algo['Title'].astype(str)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings using Sentence Transformers
embeddingsspec = model.encode(dfspec_algo['Title'].tolist(), show_progress_bar=True)
dfspec_algo['Embeddings'] = list(embeddingsspec)

# Function to save clusters to a text file
def save_clusters_to_file(df, cluster_column, filename):
    unique_clusters = sorted(df[cluster_column].unique())
    with open(filename, 'w') as file:
        for cluster in unique_clusters:
            cluster_data = df[df[cluster_column] == cluster]['Title'].tolist()
            file.write(f"Cluster {cluster}:\n")
            for title in cluster_data:
                file.write(f"    {title}\n")
            file.write("\n")

# List of different cluster sizes to try
n_clusters_list = [10, 20, 30, 40, 50, 60, 71]

# Loop through the different cluster sizes
for n_clusters in n_clusters_list:
    agg_clustering_llj = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='cosine', 
        linkage='average',
    )
    
    # Perform clustering
    dfspec_algo['Clusters'] = agg_clustering_llj.fit_predict(embeddingsspec)
    
    # Save clusters to a text file
    file_name = f"AggClusteringSpec_{n_clusters}.txt"
    save_clusters_to_file(dfspec_algo, 'Clusters', file_name)
    print(f"Clusters have been saved to {file_name}")
    
    # Add Cluster column with 'Cluster X' format
    dfspec_algo['Cluster'] = 'Cluster ' + dfspec_algo['Clusters'].astype(str)

