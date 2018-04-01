import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import numpy as np

sns.set_style('ticks')
years = range(1900, 2000, 10)

def plot_change():
	svd_change = pd.read_csv('saved/svd_change.csv').set_index('word')
	skipgram_change = pd.read_csv('saved/skipgram_change.csv').set_index('word')
	count_change = pd.read_csv('saved/count_change.csv').set_index('word')

	plt.plot(years, svd_change.ix['avg'].tolist(), 'b-', label='SVD')
	plt.plot(years, skipgram_change.ix['avg'].tolist(), 'g-', label='skipgram')
	plt.plot(years, count_change.ix['avg'].tolist(), 'r-', label='ground truth')
	plt.ylabel('Avg. Cumulative Change from 1900')
	plt.xlabel('Decade')
	plt.xlim(1900, 1990)
	plt.ylim(0, 1.0)
	plt.legend(loc=2)
	plt.xticks(years, map(lambda s: s + "s", map(str, years)))

	plt.tight_layout()
	plt.savefig('figures/compare.png')
	plt.show()

def plot_hist():
	count_svd = pickle.load(open('saved/count_svd.p'))
	count_skip = pickle.load(open('saved/count_skip.p'))
	svd_skip = pickle.load(open('saved/svd_skip.p'))

	sns.distplot(count_skip, color='b', kde=False, bins=np.arange(0, 1.0, 0.015), label='SVD')
	sns.distplot(count_svd, color='g', kde=False, bins=np.arange(0, 1.0, 0.015), label='skipgram')
	print np.mean(count_skip), np.mean(count_svd)

	plt.xlabel('Cosine Similarity with Ground Truth Vector')
	plt.ylabel('Number of Words')
	plt.legend(loc='best')
	plt.xlim(0, 1.0)
	plt.tight_layout()

	plt.savefig('figures/hist.png')
	plt.show()
	plt.close()
