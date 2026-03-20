import json
import os
import MeCab
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

directory = "./wikiextractor/extracted"

def main():
	sentences = word2vec.Text9Corpus("test.txt")
	print(list(sentences))
	# estimated_time = estimateTime()
	# sentences=SentenceIterator(estimated_time)
	# model = Word2Vec(
	# 	sentences,
	# 	vector_size=200,   # embedding size
	# 	window=5,          # context window
	# 	min_count=5,       # ignore rare words
	# 	workers=4,         # CPU cores
	# 	epochs=5           # training iterations
	# )
	# model.save("word2vec.model")

if __name__ == "__main__":
	main()
