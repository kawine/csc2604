from nltk import word_tokenize, ngrams
import pandas as pd
from collections import Counter
import csv
import sys
import string

reload(sys)
sys.setdefaultencoding('utf8')

tokens = word_tokenize(open('moby_dick.txt').read())
tokens = filter(lambda t: t not in string.punctuation, tokens)

NGRAM_COUNTS = {}

for n in range(1,4):
	with open('csv/{0}gram.csv'.format(n), 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['ngram', 'count'])
		counter = Counter(ngrams(tokens, n))

		for gram in counter:
			writer.writerow([' '.join(gram), counter[gram]])
			
		NGRAM_COUNTS[n] = pd.read_csv('csv/{0}gram.csv'.format(n)).set_index('ngram')


def sample(order, prev):
	grams = NGRAM_COUNTS[order].select(lambda i: i.startswith(prev)) 
	return grams.sample(weights='count').index[0]


def generate(order):
	sentence = []
	
	for j in range(10):
		if order == 1:
			prev = ''
		else:
			prev = ' '.join(sentence[-(order - 1):] + [''])

		sentence.append(sample(min(order, len(sentence) + 1), prev).split(' ')[-1]) 
		print len(sentence), ' '.join(sentence)

	return sentence
			

if __name__ == "__main__":
	for order in range(1,4):
		print "ORDER {0}\n".format(order)

		for i in range(5):
			generate(order)
			print

		print 
