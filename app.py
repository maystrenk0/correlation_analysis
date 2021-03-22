import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('suicide.csv')
df = df.dropna()
sns.set(style="ticks")
sns.pairplot(df)
plt.show()
sns.set(style="white")
corr = df.corr()
f, ax = plt.subplots()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.3g')
plt.show()
print('pearson')
print(df.corr(method='pearson'))
print('\n')
print('spearman')
print(df.corr(method='spearman'))
print('\n')
print('kendall')
print(df.corr(method='kendall'))
print('\n')
print(stats.pearsonr(np.array(df['HDI for year']), np.array(df['year'])))
print(stats.spearmanr(np.array(df['HDI for year']), np.array(df['year'])))
print(stats.kendalltau(np.array(df['HDI for year']), np.array(df['year'])))
