import sys
import os
import gensim
reload(sys)
sys.setdefaultencoding("utf-8")

corpora = {}

# build training corpora for each year
for i in range(800):
	fn = 'csv/googlebooks-eng-all-5gram-20090715-{0}.csv'.format(i)
	print fn

	for line in open(fn):
		text, year, count, _, _ = line.split('\t')
		year = int(year)
			
		if 1800 <= year < 2000 and year % 20 == 0:
			if year not in corpora:
				corpora[year] = open('corpora/{0}.txt'.format(year), 'w')

			for j in range(int(count)):
				corpora[year].write(text + '\n')

# train and save models
for year in corpora:
	corpora[year].close()
	sentences = gensim.models.word2vec.LineSentence('corpora/{0}.txt'.format(year))	
	model = gensim.models.word2vec.Word2Vec(sentences, window=5, min_count=1)
	model.save_word2vec_format('models/{0}.bin', binary=True)
	
