from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
# Draw Plot
fig = plt.figure()
plt.subplot(231)
df['value'].plot()
plt.title('original')

# After differentiating the sequence
df['value_diff'] = df['value'] - df['value'].shift(1)
plt.subplot(232)
df['value_diff'].dropna().plot()
plt.title('differentiating the sequence')

#After seasonal difference
n = 7
df['value_diff'] = df['value'] - df['value'].shift(n)
plt.subplot(233)
df['value_diff'].dropna().plot()
plt.title('differentiating the season')

# White Noise: Streng Stationar
randvals = np.random.randn(1000)
plt.subplot(235)
pd.Series(randvals).plot(title='Random White Noise', color='k')
plt.title("White Noise")

# transform
df['value_log'] = np.log(df['value'])
df['value_log_diff'] = df['value_log'] - df['value_log'].shift(1)
plt.subplot(234)
df['value_log_diff'].dropna().plot()
plt.title('after transform')
plt.show()

# ADF Test
result = adfuller(df.value.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
result = kpss(df.value.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


# AKF Draw Plot
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
autocorrelation_plot(df.value.tolist())
plt.show()

