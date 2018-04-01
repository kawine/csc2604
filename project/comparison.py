from scipy.spatial.distance import cosine, euclidean
from gensim.models.keyedvectors import KeyedVectors
import cPickle as pickle
import numpy as np
import heapq
import pandas as pd
import gensim
from scipy.stats import ttest_ind, ttest_1samp

years = range(1900, 2000, 10) 
target = list(set([ line.split()[2] for line in open('lemma.al') ]))

skipgram = {}
svd_vocab = {}
svd_vectors = {}

for year in years:
	skipgram[year] = KeyedVectors.load_word2vec_format('sgns/{0}.bin'.format(year), binary=True)

	svd_vocab[year] = np.load('svd/{0}-vocab.pkl'.format(year))
	svd_vectors[year] = np.load('svd/{0}-w.npy'.format(year))

count = pd.read_csv('count_decade.csv').set_index('word')


def get_vector(word, year, typ):
	if typ == 'skipgram':
		return skipgram[year][word]
	elif typ == 'svd':
		return svd_vectors[year][svd_vocab[year].index(word)]
	else:
		raise Exception


def compare(w, y1, y2, t='skipgram', similarity=False):
	try:
		dist = cosine(get_vector(w, y1, t), get_vector(w, y2, t))
		dist = 1 - dist if similarity else dist
		return np.nan_to_num(dist)
	except Exception:
		return 0


def changing(m=30, inverse=False, t='skipgram'):
	"""
	Return the top m most(least) changing words if inverse is False(True).
	"""
	f = lambda w: compare(w, 1900, 1990, t=t, similarity=inverse)
	words = heapq.nlargest(m, target, key=f)
	return zip(words, map(f, words))


def hyp_test():
	"""
	Determine if there is a significant difference between most and least changing words.
	"""
	x = map(lambda w: count.ix[w].sum(), zip(*changing(inverse=False))[0])
	y = map(lambda w: count.ix[w].sum(), zip(*changing(inverse=True))[0])
	print np.mean(x), np.mean(y), ttest_ind(x, y, equal_var=True)


def cosine_test():
	"""
	Create a smoothed ground-truth vector for each word by using the aggregate change in
	word senses since 1900. Compare this to the basline (SVD of a PPMI vector from 
	Hamilton et al.) and the proposed model.
	"""
	# ground-truth
	count_change = count.copy()

	for year in list(count_change): 
		for prev in range(1900, int(year), 10):
			count_change[year] += count[str(prev)]

	svd_change = count_change.copy()	# baseline
	skipgram_change = count_change.copy()	# proposed approach

	for word in count.index:
		for year in list(count_change):
			svd_change.loc[word, year] = compare(word, 1900, int(year), t='svd')
			skipgram_change.loc[word, year] = compare(word, 1900, int(year), t='skipgram')

	count_svd, count_skip, svd_skip = [], [], []

	for w in count.index:
		if count_change.ix[w].sum() == 0:
			count_change.ix[w] += 1

		count_change.ix[w] /= float(count_change.loc[w,'1990'] + 0.00001)

		count_svd.append(1 - cosine(count_change.ix[w], svd_change.ix[w]))
		count_skip.append(1 - cosine(count_change.ix[w], skipgram_change.ix[w]))
		svd_skip.append(1 - cosine(svd_change.ix[w], skipgram_change.ix[w]))

	print "svd/count vs. skip/count", ttest_ind(np.nan_to_num(count_svd), np.nan_to_num(count_skip), equal_var=True)
	print "svd/skip", ttest_1samp(np.nan_to_num(svd_skip), 0.95)
	print np.mean(np.nan_to_num(count_svd)), np.mean(np.nan_to_num(count_skip)), np.mean(svd_skip)

	svd_change.ix['avg'] = svd_change.mean(axis=0)
	skipgram_change.ix['avg'] = skipgram_change.mean(axis=0)
	count_change.ix['avg'] = count_change.mean(axis=0)

	svd_change.to_csv('saved/svd_change.csv')
	skipgram_change.to_csv('saved/skipgram_change.csv')
	count_change.to_csv('saved/count_change.csv')
	
	pickle.dump(count_svd, open('saved/count_svd.p', 'w'))
	pickle.dump(count_skip, open('saved/count_skip.p', 'w'))
	pickle.dump(svd_skip, open('saved/svd_skip.p', 'w'))		


def tabulate(m=30):
	"""
	Return a pandas DataFrame with the changes in the most(least) changing words if inverse is
	False(True).
	"""
	for inverse, kind in [(False, 'most'), (True, 'least')]:
		words = zip(*changing(m=m, inverse=inverse))[0]
		dist = []
		similar_words = []

		for w in words:
			dist.append([w] + map(lambda y: round(compare(w, 1900, y, t='skipgram'), 2), years))
			similar_words.append([w] + [ ', '.join(most_similar(w, 1900)), ', '.join(most_similar(w, 1990)) ]) 

		df = pd.DataFrame(dist, columns=(['word'] + years)).set_index('word')
		df.ix['avg'] = df.mean(axis=0)
		df.to_csv('saved/{0}_changing_dist.csv'.format(kind) , sep='&')

		df = pd.DataFrame(similar_words, columns=['word', 'start (1900)', 'end (1990)']).set_index('word')
		df.to_csv('saved/{0}_changing_similar.csv'.format(kind), sep='&')


def most_similar(word, year, m=4, t='skipgram'):
	"""
	Return the m most similar words for the specified word in the specified year.
	"""
	def f(w): 
		try:
			return np.nan_to_num(1 - cosine(get_vector(word, year, t), get_vector(w, year, t)))
		except Exception:
			return 0	

	top_m = heapq.nlargest(m, target, key=f)
	top_m.remove(word)

	return top_m


