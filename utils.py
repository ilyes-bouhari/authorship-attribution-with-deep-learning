import os
import numpy as np
import tensorflow as tf
import pickle
import gensim.downloader
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten


def normlize(text):

    import re
    import string
    import nltk
    import contractions

    nltk.download("stopwords")
    from nltk.corpus import stopwords

    result = tf.strings.lower(text)
    result = tf.strings.regex_replace(result, "<[^>]+>", " ")
    result = tf.strings.regex_replace(result, "\r\n", " ")
    result = tf.strings.regex_replace(
        result, "https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", ""
    )

    for index, (key, value) in enumerate(contractions.contractions_dict.items()):
        result = tf.strings.regex_replace(result, r"\b(" + key + r")\b", value)

    result = tf.strings.regex_replace(
        result, r"\b(" + r"|".join(stopwords.words("english")) + r")\b\s*", ""
    )
    result = tf.strings.regex_replace(result, f"[{re.escape(string.punctuation)}]", "")

    return result


def load_glove_embeddings(path, embedding_dim):

    path_to_glove_file = os.path.join(
        os.path.expanduser("~"), "{}/glove.6B.{}d.txt".format(path, embedding_dim)
    )

    embeddings_index = {}
    f = open(path_to_glove_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    print("Loaded %s word vectors." % len(embeddings_index))

    return embeddings_index


def create_embeddings_matrix(embeddings_index, vocabulary, embedding_dim=100):

    embeddings_matrix = np.random.rand(
        len(vocabulary) + 1, embeddings_index.vector_size
    )

    for i, word in enumerate(vocabulary):

        try:
            embedding_vector = embeddings_index.get_vector(word)
        except:
            embedding_vector = None

        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    print("Matrix shape: {}".format(embeddings_matrix.shape))

    return embeddings_matrix


def get_embeddings_layer(embeddings_matrix, name, max_len, trainable=False):

    embedding_layer = Embedding(
        input_dim=embeddings_matrix.shape[0],
        output_dim=embeddings_matrix.shape[1],
        input_length=max_len,
        weights=[embeddings_matrix],
        trainable=trainable,
        name=name,
    )

    return embedding_layer


def get_conv_pool(x_input, max_len, suffix, n_grams=[3, 4, 5], feature_maps=100):

    branches = []
    for n in n_grams:

        branch = Conv1D(
            filters=feature_maps,
            kernel_size=n,
            activation="relu",
            name="Conv_" + suffix + "_" + str(n),
        )(x_input)

        branch = MaxPooling1D(
            pool_size=max_len - n + 1,
            strides=None,
            padding="valid",
            name="MaxPooling_" + suffix + "_" + str(n),
        )(branch)

        branch = Flatten(name="Flatten_" + suffix + "_" + str(n))(branch)

        branches.append(branch)

    return branches


def get_dataset_partitions_tf(
    ds,
    ds_size,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    shuffle=True,
    shuffle_size=10000,
):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def get_text_vectorizer(max_features, output_mode, raw_train_ds, sequence_length):
    import pickle
    import os.path
    import tensorflow as tf

    if output_mode == "tf_idf":
        sequence_length = None

    filename = "{output_mode}_{max_features}_{sequence_length}.pkl".format(
        output_mode=output_mode,
        max_features=max_features,
        sequence_length=sequence_length,
    )

    if not os.path.exists(filename):

        vectorizer = tf.keras.layers.TextVectorization(
            standardize=normlize,
            max_tokens=max_features,
            output_sequence_length=sequence_length,
            output_mode=output_mode,
        )

        with tf.device("CPU"):
            text_ds = raw_train_ds.map(lambda x, y: x)
            vectorizer.adapt(text_ds)

        pickle.dump(
            {"config": vectorizer.get_config(), "weights": vectorizer.get_weights()},
            open(filename, "wb"),
        )

    else:

        from_disk = pickle.load(open(filename, "rb"))
        vectorizer = tf.keras.layers.TextVectorization.from_config(from_disk["config"])
        vectorizer.set_weights(from_disk["weights"])

    return vectorizer


def vectorize_text(text, label, vectorizer):
    text = tf.expand_dims(text, -1)
    return vectorizer(text), label


def load_pre_trained_embeddings():

    pre_trained_embeddings = {
        # 'word2vec-google-news-300': 'word2vec_300',
        "glove-wiki-gigaword-50": "glove_50",
        # 'glove-wiki-gigaword-100': 'glove_100',
        # 'glove-wiki-gigaword-200': 'glove_200',
        # 'glove-wiki-gigaword-300': 'glove_300',
    }

    for (model, filename) in pre_trained_embeddings.items():
        pickle.dump(
            gensim.downloader.load(model),
            open("{filename}.pkl".format(filename=filename), "wb"),
        )


def load_file(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    return data
