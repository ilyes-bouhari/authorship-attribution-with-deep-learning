import os
import numpy as np
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten

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

def get_embeddings_layer(embeddings_matrix, name, max_len, trainable=False):
    
    embedding_layer = Embedding(
        input_dim=embeddings_matrix.shape[0],
        output_dim=embeddings_matrix.shape[1],
        input_length=max_len,
        weights=[embeddings_matrix],
        trainable=trainable,
        name=name
    )
        
    return embedding_layer

def get_conv_pool(x_input, max_len, suffix, n_grams=[3, 4, 5], feature_maps=100):
    
    branches = []
    for n in n_grams:

        branch = Conv1D(
            filters=feature_maps, 
            kernel_size=n, 
            activation='relu',
            name='Conv_' + suffix + '_' + str(n)
        )(x_input)

        branch = MaxPooling1D(
            pool_size=max_len - n + 1, 
            strides=None, 
            padding='valid',
            name='MaxPooling_' + suffix + '_' + str(n)
        )(branch)
        
        branch = Flatten(name='Flatten_' + suffix + '_' + str(n))(branch)

        branches.append(branch)

    return branches