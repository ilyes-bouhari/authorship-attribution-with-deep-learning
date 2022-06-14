import os
import numpy as np

def load_glove_embeddings(path, embedding_dim):

    path_to_glove_file = os.path.join(
        os.path.expanduser("~"), "{}/glove.6B.{}d.txt".format(path, embedding_dim)
    )

    embeddings_index = {}
    f = open(path_to_glove_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
  
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def create_embeddings_matrix(embeddings_index, vocabulary, embedding_dim=100):

    embeddings_matrix = np.random.rand(len(vocabulary) + 1, embedding_dim)
    for i, word in enumerate(vocabulary):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    print('Matrix shape: {}'.format(embeddings_matrix.shape))
    return embeddings_matrix