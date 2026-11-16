import numpy as np
import configurations

SHAPE = configurations.SHAPE


def embed(config):
    embedding = np.zeros(np.sum(SHAPE))
    for i in range(len(SHAPE)):
        embedding[sum(SHAPE[:i])+config[i]] = 1
    return embedding

def embed_all(N_CONFIGS = configurations.N_CONFIGS):
    embeddings = np.zeros((N_CONFIGS, np.sum(SHAPE)))
    for i in range(N_CONFIGS):
        embeddings[i, :] = embed(configurations.decode(i))
    return embeddings

def distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    n = embeddings.shape[1]  # embeddings is (20, N_CONFIGS)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(embeddings[i, :] - embeddings[j, :])
    return dist
    

