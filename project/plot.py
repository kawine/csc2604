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

	fig, ax1 = plt.subplots()
	
	ax1.plot(years, svd_change.ix['avg'].tolist(), 'b-', label='SVD')
	ax1.plot(years, skipgram_change.ix['avg'].tolist(), 'b--', label='skipgram')
	ax1.set_ylabel('Avg. Cosine Distance from Start', color='b')
	ax1.set_xlabel('Year')
	ax1.set_xlim(1900, 1990)
	ax1.set_ylim(0, 1.0)
	plt.legend(loc=2)

	ax2 = ax1.twinx()
	ax2.plot(years, count_change.ix['avg'].tolist(), 'r-', label='sense count')
	ax2.set_ylabel('Avg. Cumulative Sense Count', color='r')
	ax2.set_xlabel('Year')
	ax2.set_xlim(1900, 1990)
	ax2.set_ylim(0, 1.0)
	plt.legend(loc=1)

	fig.tight_layout()
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
