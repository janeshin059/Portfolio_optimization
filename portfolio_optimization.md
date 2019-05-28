
# Portfolio Optimization

1. original mean-variance model
2. mean-variance-skewness-kurtosis model
3. feature scaling
4. optimized mean-variance-skewness-kurtosis model(with new feature)
5. result
6. conclusion


## 1. Original mean-variance model
### The Efficient Frontier : Markowitz portfolio optimization 

Modern Portfolio Theory is about how investors construct portfolios that maximizes returns for given
levels of risk.


1.마코비츠의 평균 분산 모형

in markowitz's mean-variance model
: 


2. 마코비츠의 평균 분산 모형의 확장
3. 평균 분산 모형을 ML알고리즘에..?





https://medium.com/python-data/effient-frontier-in-python-34b0c3043314
https://medium.com/python-data/efficient-frontier-portfolio-optimization-with-python-part-2-2-2fe23413ad94


```python
# get data first from Quandl

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import quandl

quandl.ApiConfig.api_key = 'RnbjVbHE64Ds4TvB51sq'
tickers = ['CNP', 'F', 'WMT', 'GE', 'TSLA']

data = quandl.get_table('WIKI/PRICES', ticker = tickers,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2009-05-01', 'lte': '2019-05-27' }, paginate=True)# 최근10년
clean = data.set_index('date')
table = clean.pivot(columns='ticker')
```


```python

```


```python
# get data from Yahoo

import fix_yahoo_finance as yf

start = dt.datetime(2009,1,1)
end = dt.datetime.now()

start = '2009-01-01' # 2008년에는 세계 금융 위기에 의해 주식시장이 정상적이지 않은 상태였으므로..
# 최근 10년  관찰
tickers = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
#tickers = '^IXIC'
data = yf.download(tickers, start = start, end = end, group_by = 'tickers')

```


```python
# import tickers from Wikipedia
######### 일단 종목 다섯개만 가져왔고, 더 가져올 때 사용
import pickle
import requests
import bs4 as bs
import lxml

web_add = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
request = requests.get(web_add)
soup = bs.BeautifulSoup(request.text,'html.parser')
table = soup.find('table',{'class': 'wikitable sortable'})
##download

#tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)
    
#data = quandl.get_table('WIKI/PRICES',ticker = tickers,  trim_start = start, trim_end = end,  authtoken = auth_tok)

```


```python
#data.head(10)
```

to get the efficient frontier, we need to simulate imaginary combinations of portfolios
 


```python
#data.columns.names = ['tickers', 'info']
#data.head(5)
```


```python
table.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">adj_close</th>
    </tr>
    <tr>
      <th>ticker</th>
      <th>CNP</th>
      <th>F</th>
      <th>GE</th>
      <th>TSLA</th>
      <th>WMT</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-05-01</th>
      <td>7.387214</td>
      <td>4.484893</td>
      <td>9.657888</td>
      <td>NaN</td>
      <td>40.359445</td>
    </tr>
    <tr>
      <th>2009-05-04</th>
      <td>7.360376</td>
      <td>4.630711</td>
      <td>9.969924</td>
      <td>NaN</td>
      <td>40.996487</td>
    </tr>
    <tr>
      <th>2009-05-05</th>
      <td>7.226185</td>
      <td>4.611006</td>
      <td>9.969924</td>
      <td>NaN</td>
      <td>40.690062</td>
    </tr>
    <tr>
      <th>2009-05-06</th>
      <td>7.273152</td>
      <td>4.934170</td>
      <td>10.403729</td>
      <td>NaN</td>
      <td>39.923999</td>
    </tr>
    <tr>
      <th>2009-05-07</th>
      <td>7.279861</td>
      <td>4.776529</td>
      <td>10.624438</td>
      <td>NaN</td>
      <td>40.230424</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## What is the optimal portfolio among the various optimal combinations??

1. sharp ration를 넣어본다(demo) 

is simply as a measure of the performance of an investment's retuens given its risk

2. kurtosis를 넣어본다    

3.skewness를 넣어본다  






```python
from IPython.display import Image
Image(filename = 'C:/Users/user/Desktop/19_1/Financial_Data_with_python/sharp.jpeg')

```




![jpeg](output_13_0.jpeg)



pct_change:수익률 계산

we should

#### portfolio중  
#### maximizing return & skewness  
#### minimizing risk &kurtosis  
#### 이런 portfolio를 선택해야 함.  

(in return distributions)
-->why??

portfolio return을 어떻게구함..?

https://flyinglightly.tistory.com/30



```python
# calculate daily and annual returns of the stocks
returns_daily = table.pct_change()
returns_annual = returns_daily.mean( ) * 250


### 질문: 보통 log returns를 많이 쓰던데(논문에서도..) 그냥 pct_change를 써도 문제가 안 발생할까??
#비슷하긴 한데...수렴하긴 하는듯...

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily*250

# skewness

skew_daily = returns_daily.skew()
skew_annual = skew_daily*250

#kurtosis

kurt_daily = returns_daily.kurtosis()
kurt_annual = kurt_daily*250

port_returns = []
port_volatility = []

port_skewness = []
port_kurtosis = []

sharpe_ratio = []
stock_weights = []


# the number of combinations for imaginary portfolios
num_assets = len(tickers)
num_portfolios = 50000


# set random seed for reproduction's sake
np.random.seed(101)


#populate the empty lists with each portfolios returns, risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets) # asset의 개수만큼의 random number 생성
    weights /= np.sum(weights)
    returns = np.dot(weights,returns_annual) #vector와 matrix의 곱
    volatility = np.sqrt(np.dot(weights.T, np.dot (cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)
    
    
    
```


```python
# a dictionary for returns and risk values of each portfolio
portfolio = {'Returns':port_returns,'Volatility':port_volatility, 'Sharpe Ratio':sharpe_ratio}

# extended original dictionary to accomadate each ticke and weight in the portfolio
for counter,symbol in enumerate(tickers):
    portfolio[symbol + 'Weight'] = [Weight[counter] for Weight in stock_weights]
    
df = pd.DataFrame(portfolio)

column_order = ['Returns','Volatility','Sharpe Ratio'] + [stock + 'Weight' for stock in tickers]

df = df[column_order]




```




```python

```
