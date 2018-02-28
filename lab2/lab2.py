from scipy.spatial.distance import cosine
import cPickle as pickle
import numpy as np
import heapq
import pandas as pd
import gensim

years = range(1800, 1990, 20) 
target = list(set([ line.split()[2] for line in open('lemma.al') ]))

vectors = {}

for year in years:
	vectors[year] = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('models/{0}.bin'.format(year), binary=True)

get_vector = lambda word, year: vectors[year][word]

def compare(t, y1, y2, similarity=False):
	try:
		dist = cosine(get_vector(t, y1), get_vector(t, y2))
		dist = 1 - dist if similarity else dist
		return np.nan_to_num(dist)
	except KeyError:
		return 0

def changing(m=30, inverse=False):
	"""
	Return the top m most(least) changing words if inverse is False(True).
	"""
	f = lambda t: compare(t, 1800, 1980, similarity=inverse)
	words = heapq.nlargest(m, target, key=f)
	return zip(words, map(f, words))

def tabulate(m=30, inverse=False):
	"""
	Return a pandas DataFrame with the changes in the most(least) changing words if inverse is
	False(True).
	"""
	words = zip(*changing(m=m, inverse=inverse))[0]
	table = []

	for w in words:
		table.append([w] + map(lambda y: round(compare(w, 1800, y), 2), years))

	df = pd.DataFrame(table, columns=(['word'] + years)).set_index('word')
	df.ix['avg'] = df.mean(axis=0)
	print df.to_csv(sep='&')
