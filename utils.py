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