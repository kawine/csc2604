from gensim.models import KeyedVectors
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import numpy as np
import analysis_e

# Step 2
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
rg65 = pd.read_csv('rg65.csv')

# Step 3
scores = np.nan_to_num([ 1 - cosine(model[a], model[b]) for i,(a,b,s) in rg65.iterrows() ])
print pearsonr(scores, rg65['score'].tolist())[0]

# Step 4
acc = model.accuracy('./word-test.v1.txt')
for section in acc: 
	print section['section'], float(len(section['correct'])) / (len(section['correct']) + len(section['incorrect'])) 


with open('M2_100.txt', 'w') as f:
	f.write("{0} {1}\n".format(len(analysis_e.word_index), 100))
	for w in analysis_e.word_index:
		f.write(w + ' ' + ' '.join(map(str, analysis_e.M2_100[analysis_e.word_index[w]])) + '\n')

lsa_model = KeyedVectors.load_word2vec_format('M2_100.txt', binary=False)
acc = lsa_model.accuracy('./word-test.v1.txt')

for section in acc: 
	if len(section['correct']) + len(section['incorrect']) > 0:
		print section['section'], float(len(section['correct'])) / (len(section['correct']) + len(section['incorrect']))
