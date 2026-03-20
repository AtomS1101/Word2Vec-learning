import json
import os
import MeCab
from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

directory = "./wikiextractor/extracted"

def estimateTime():
	estimated_time = 0
	for root, dirs, files in os.walk(directory):
		estimated_time += len(files)
	return estimated_time

class SentenceIterator:
	def __init__(self, estimated_time):
		self.estimated_time = estimated_time
		self.count = 0
		self.mecab = MeCab.Tagger("-Owakati")

	def __iter__(self):
		for root, dirs, files in os.walk(directory):
			print(f'Processing "{root}", {self.count}/{self.estimated_time}')
			for file in files:
				if not file.startswith("wiki"):
					continue
				path = os.path.join(root, file)
				with open(path, encoding="utf-8") as f:
					for line in f:
						json_data = json.loads(line)
						text = json_data['text']
						if not text:
							continue
						sentence = []
						node = self.mecab.parseToNode(text)
						node = node.next
						while node:
							word = node.surface
							parts = node.feature.split(",")[0]
							if word == "。" and sentence:
								yield sentence
								sentence = []
							else:
								sentence.append(word)
							node = node.next
				self.count += 1

def main():
	estimated_time = estimateTime()
	sentences=SentenceIterator(estimated_time)
	model = Word2Vec(
		sentences,
		vector_size=200,   # embedding size
		window=5,          # context window
		min_count=5,       # ignore rare words
		workers=4,         # CPU cores
		epochs=5           # training iterations
	)
	model.save("word2vec.model")

if __name__ == "__main__":
	main()
