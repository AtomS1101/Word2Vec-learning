import gensim
model = gensim.models.Word2Vec.load('vec200win15min20epo5(NounVec)/word2vec.model')
# model = gensim.models.Word2Vec.load('vec200win5min5epo5(all)/word2vec.model')

positive = "テレビ"
negative = "画像"
count = 10


print(f"\n( {' + '.join(positive.split())}{' - ' if negative else ''}{' - '.join(negative.split())} ) =")
result = model.wv.most_similar(positive=positive.split(), negative=negative.split(), topn=count)
for word, score in result:
	space = "\u3000" * (20 - len(word))
	print(f"{word}{space}: {score}")
