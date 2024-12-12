import gensim
word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)

# слова «тракт» и «долина»

w1 = word2vec["горный_ADJ"]
w2 = word2vec["дорога_NOUN"]

result = w1 + w2
dist = word2vec.similar_by_vector(result, topn=10)
for i in dist:
  print(i)