from nltk.corpus import brown
from nltk import FreqDist
from nltk import ngrams
import nltk
from collections import Counter
import numpy as np
from scipy.linalg import svd
from scipy.stats import threshold
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import pandas as pd

rg65 = pd.read_csv('rg65.csv')


# Step 1 : download corpus
nltk.download('brown')


# Step 2: unigram frequencies
N = 5000
fdist = FreqDist(map(lambda w: w.lower(), brown.words()))
W = fdist.most_common(N)

# add n additional words not in top 5000
for w in set(rg65['w1'].tolist() + rg65['w2'].tolist()) - set(zip(*W)[0]):
	W.append((w, fdist.freq(w) * fdist.N()))

word_index = { v[0] : i for i,v in enumerate(W) }
N = len(word_index)


# Step 3: word-context matrix
M1 = np.zeros((N, N)) 
words = brown.words()

for i in xrange(len(words)):
	if i+1 < len(words) and words[i] in word_index and words[i+1] in word_index:
		M1[word_index[words[i]], word_index[words[i+1]]] += 1

	if i-1 > 0 and words[i] in word_index and words[i-1] in word_index:
		M1[word_index[words[i]], word_index[words[i-1]]] += 1

# Step 4: PPMI matrix
p_wc = M1 / M1.sum()
p_w = M1.sum(axis=0) / M1.sum()
p_c = M1.sum(axis=1) / M1.sum()
M1_plus = np.nan_to_num(threshold(np.log(p_wc / np.outer(p_w, p_c)), threshmin=0))


# Step 5: SVD
U, s, V = svd(M1_plus)
M2_10 = U[:,:10]
M2_50 = U[:,:50]
M2_100 = U[:,:100]


# Step 6: human scores
S = rg65

# Step 7: model scores
S_hat = {}
models = ['M1', 'M1_plus', 'M2_10', 'M2_50', 'M2_100']

for model in models:
	M = eval(model)
	S_hat[model] = np.nan_to_num([ 1 - cosine(M[word_index[a],:], M[word_index[b],:]) for i,(a,b,_) in S.iterrows() ])


# Step 8: pearson R
for model in models:
	print model, np.nan_to_num(pearsonr(S['score'].tolist(), S_hat[model])[0])	
