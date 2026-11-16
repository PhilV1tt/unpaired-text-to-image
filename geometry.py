import numpy as np
import configurations

SHAPE = configurations.SHAPE


def embed(config, SHAPE):
    embedding = np.zeros(np.sum(SHAPE))
    avancement = 0
    for i in range(len(SHAPE)):
        embedding[sum(SHAPE[:i])+config[i]] = 1
    return embedding

def embed_all(N_CONFIGS = configurations.N_CONFIGS):
    embeddings = np.zeros((20, N_CONFIGS))
    for i in range(N_CONFIGS):
        embeddings[:, i] = embed(configurations.decode(i), SHAPE)
    return embeddings

def distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    n = embeddings.shape[1]  # embeddings is (20, N_CONFIGS)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(embeddings[:, i] - embeddings[:, j])
    return dist
    

