import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import hdbscan
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# %%
dfllj_hdb = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dflljyear.csv')
dfspec_hdb = pd.read_csv('/Users/matt/Desktop/LLJ-Data/LLJSpectrum/YearAlteredData/dfspecyear.csv')

# %%
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(tokens)
    return text

# %%
dfllj_hdb['Title'] = dfllj_hdb['Title'].apply(preprocess_text)
dfspec_hdb['Title'] = dfspec_hdb['Title'].apply(preprocess_text)

# %%
dfllj_hdb['Title'] = dfllj_hdb['Title'].astype(str)
embeddingsllj = model.encode(dfllj_hdb['Title'].tolist())
dfllj_hdb['Embeddings'] = list(embeddingsllj)

# %%
dfspec_hdb['Title'] = dfspec_hdb['Title'].astype(str)
embeddingsspec = model.encode(dfspec_hdb['Title'].tolist())
dfspec_hdb['Embeddings'] = list(embeddingsspec)


pca = PCA(n_components=50)  
reduced_embeddings_llj = pca.fit_transform(embeddingsllj)

# %%
embeddings_array_llj = np.array(reduced_embeddings_llj)
clustererhdb = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric='euclidean')
cluster_labels_llj = clustererhdb.fit_predict(embeddings_array_llj)
dfllj_hdb['Cluster'] = cluster_labels_llj

# %%
num_clusters = len(set(cluster_labels_llj)) - (1 if -1 in cluster_labels_llj else 0)
print(f'Number of clusters: {num_clusters}')

# %%
for cluster in set(cluster_labels_llj):
    if cluster != -1:  # -1 is the noise label
        print(f'\nCluster {cluster}:')
        cluster_titles = dfllj_hdb[dfllj_hdb['Cluster'] == cluster]['Title'].tolist()
        for title in cluster_titles:
            print(title)

# %%



