
# Portfolio Optimization



### The Efficient Frontier : Markowitz portfolio optimization 

Modern Portfolio Theory is about how investors construct portfolios that maximizes returns for given
levels of risk.

#### get data first from Quandl


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import quandl

quandl.ApiConfig.api_key = ''
tickers = ['CNP', 'F', 'WMT', 'GE']
## change tickers
data = quandl.get_table('WIKI/PRICES', ticker = tickers,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2009-05-01', 'lte': '2019-05-27' }, paginate=True)# recent 10 years
clean = data.set_index('date')
table = clean.pivot(columns='ticker')

table.head(5)
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
      <th colspan="4" halign="left">adj_close</th>
    </tr>
    <tr>
      <th>ticker</th>
      <th>CNP</th>
      <th>F</th>
      <th>GE</th>
      <th>WMT</th>
    </tr>
    <tr>
      <th>date</th>
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
      <td>40.359445</td>
    </tr>
    <tr>
      <th>2009-05-04</th>
      <td>7.360376</td>
      <td>4.630711</td>
      <td>9.969924</td>
      <td>40.996487</td>
    </tr>
    <tr>
      <th>2009-05-05</th>
      <td>7.226185</td>
      <td>4.611006</td>
      <td>9.969924</td>
      <td>40.690062</td>
    </tr>
    <tr>
      <th>2009-05-06</th>
      <td>7.273152</td>
      <td>4.934170</td>
      <td>10.403729</td>
      <td>39.923999</td>
    </tr>
    <tr>
      <th>2009-05-07</th>
      <td>7.279861</td>
      <td>4.776529</td>
      <td>10.624438</td>
      <td>40.230424</td>
    </tr>
  </tbody>
</table>
</div>




```python
(table/table.iloc[0] *100).plot(figsize = (10,6),grid = True)
plt.show()
```


![output_3_0](https://user-images.githubusercontent.com/41497195/58821545-472b7400-8670-11e9-891f-5cb115887997.png)


## What is the optimal portfolio among the various optimal combinations?
to get the efficient frontier, we need to simulate imaginary combinations of portfolios.  
we should select portfolio which has maximized return & skewness , as well as minimized risk &kurtosis.  

the paramont interest to investors is to find what risk-return profiles are possible for a given set of financial instruments, and their statistical characertistics.

to make a list of situations, Monte Carlo simulation for generating random portfolio weight vectors is required.  
first, we should populate the empty lists with each portfolios returns, risk and weights.  
then, we will get expected portfolio return and variance in each simulated allocation.


### the PGP(polynomial goal programming) model

![pgp](https://user-images.githubusercontent.com/41497195/58840491-ccc71800-86a0-11e9-9802-29d1312bc7ae.PNG)

Z: the objective function  

X: weights(asset allocation rate) vector  

d1,d2,d3,d4 : deviations of expected return, variance, skewness  from the optimal scores of, R*, V*, S* and K*, respectively  

lambda 1,2,3,4 : preferences of investor  


```python
# calculate daily and annual returns of the stocks
returns_daily = table.pct_change()
returns_annual = returns_daily.mean( ) * 250

n_returns_daily = pd.DataFrame(returns_daily)

df2 = pd.DataFrame(np.round(np.random.randn(2241, 4), 2),columns = [['adj_close','adj_close','adj_close','adj_close'],['CNP','F','GE','WMT']])

df2.index.names  = ['ticker']
df2['adj_close','CNP'] = 0.000654
df2['adj_close','F'] = 0.000576
df2['adj_close','GE']  = 0.000257
df2['adj_close','WMT'] = 0.000395


def returns(weights):
    return np.dot(weights,returns_annual)

def volatility(weights):
    return  np.sqrt(np.dot(weights.T, np.dot (cov_annual, weights)))


def kurtosis(weights):
    return (np.dot(weights.T,(n_returns_daily.reset_index(drop=True)).sub(df2).fillna(0).T).sum()/len(tickers))**4

def skewness(weights): 
    return  (np.dot(weights.T,(n_returns_daily.reset_index(drop=True)).sub(df2).fillna(0).T).sum()/len(tickers))**3

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
num_portfolios = 10000


# set random seed for reproduction's sake
np.random.seed(101)

```


```python
returns_daily.head()
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
      <th colspan="4" halign="left">adj_close</th>
    </tr>
    <tr>
      <th>ticker</th>
      <th>CNP</th>
      <th>F</th>
      <th>GE</th>
      <th>WMT</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-05-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2009-05-04</th>
      <td>-0.003633</td>
      <td>0.032513</td>
      <td>0.032309</td>
      <td>0.015784</td>
    </tr>
    <tr>
      <th>2009-05-05</th>
      <td>-0.018232</td>
      <td>-0.004255</td>
      <td>0.000000</td>
      <td>-0.007474</td>
    </tr>
    <tr>
      <th>2009-05-06</th>
      <td>0.006500</td>
      <td>0.070085</td>
      <td>0.043511</td>
      <td>-0.018827</td>
    </tr>
    <tr>
      <th>2009-05-07</th>
      <td>0.000923</td>
      <td>-0.031949</td>
      <td>0.021214</td>
      <td>0.007675</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.head()

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
      <th colspan="4" halign="left">adj_close</th>
    </tr>
    <tr>
      <th></th>
      <th>CNP</th>
      <th>F</th>
      <th>GE</th>
      <th>WMT</th>
    </tr>
    <tr>
      <th>ticker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000654</td>
      <td>0.000576</td>
      <td>0.000257</td>
      <td>0.000395</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000654</td>
      <td>0.000576</td>
      <td>0.000257</td>
      <td>0.000395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000654</td>
      <td>0.000576</td>
      <td>0.000257</td>
      <td>0.000395</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000654</td>
      <td>0.000576</td>
      <td>0.000257</td>
      <td>0.000395</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000654</td>
      <td>0.000576</td>
      <td>0.000257</td>
      <td>0.000395</td>
    </tr>
  </tbody>
</table>
</div>




```python
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets) # make random numbers 
    weights /= np.sum(weights) # random portfolio weights
    
    p_returns = np.dot(weights,returns_annual) #product of vector and matrix
    
    p_volatility = np.sqrt(np.dot(weights.T, np.dot (cov_annual, weights)))
    
    sharpe = p_returns / p_volatility
    
    sharpe_ratio.append(sharpe)
    
    port_returns.append(returns(weights))
    
    port_volatility.append(volatility(weights))
    
    port_skewness.append(skewness(weights))
    
    port_kurtosis.append(kurtosis(weights))
    
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
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(11, 8), grid=True)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()

```


![output_12_0](https://user-images.githubusercontent.com/41497195/58821547-47c40a80-8670-11e9-85ff-909ab7d0093a.png)


# Optimal Portfolios


We will test 3 methods:
1. minimize portfolio variance

2. equal weights

3. maximizing return(mean) & skewness , minimizing risk(variance) &kurtosis
- this model is extended from Markowitz's mean-variance model(in above)


```python
import scipy.optimize as sco
```


```python
cons = ({'type':'eq','fun' : lambda x : np.sum(x) - 1}) #equality constraint

bnds = tuple((0,1) for x in range(num_assets))  #bonds for the parameters

eweights = np.array(num_assets* [1. / num_assets,]) #equal weights vector
eweights



```




    array([0.25, 0.25, 0.25, 0.25])



### minimize portfolio variance
- minimum volatility
- minimum variance portfolio



```python
optv = sco.minimize(volatility, eweights, method = 'SLSQP', bounds = bnds, constraints = cons)

optv

#this minimize fiunction is quite genral and allows for equality constaraints,
#inequility constraints, and numerical bounds for the parameters

```




         fun: 0.14076467102770465
         jac: array([0.14145519, 0.14099785, 0.14006851, 0.14048788])
     message: 'Optimization terminated successfully.'
        nfev: 31
         nit: 5
        njev: 5
      status: 0
     success: True
           x: array([0.32602012, 0.02572137, 0.12348349, 0.52477502])




```python
optv['x'].round(4)
```




    array([0.326 , 0.0257, 0.1235, 0.5248])




```python
volatility(optv['x']).round(4)


```




    0.1408




```python
returns(optv['x']).round(4)
```




    0.1168



### Equal weights




```python
cons = ({'type':'eq','fun' : lambda x : np.sum(x) - 1}) #equality constraint

bnds = tuple((0,1) for x in range(num_assets))  #bonds for the parameters

equal_weights = np.array(num_assets* [1. / num_assets,]) #equal weights vector
equal_weights
```




    array([0.25, 0.25, 0.25, 0.25])




```python
volatility(equal_weights).round(4)
```




    0.1636




```python
returns(equal_weights).round(4)
```




    0.1177



### maximizing return(mean) & skewness , minimizing risk(variance) &kurtosis

- extended Markowitz's model






```python
cons = ({'type':'eq','fun' : lambda x : np.sum(x) - 1}) #equality constraint

bnds = tuple([0,1] for x in range(num_assets))  #bonds for the parameters
```


```python
rn = lambda weights : returns(weights)

optm = sco.minimize(lambda weights : -rn(weights), eweights, method = 'SLSQP', bounds =bnds, constraints = cons)

optm
```




         fun: -0.16360798616241193
         jac: array([-0.16360799, -0.14406324, -0.06436619, -0.0987836 ])
     message: 'Optimization terminated successfully.'
        nfev: 54
         nit: 9
        njev: 9
      status: 0
     success: True
           x: array([1.00000000e+00, 3.88578059e-16, 1.89084859e-16, 0.00000000e+00])




```python
ske = lambda weights : skewness(weights)
opts = sco.minimize(lambda weights : -ske(weights), eweights, method = 'trust-constr', bounds = bnds, constraints = cons, options ={'xtol': 1e-17, 'gtol': 1e-14, 'barrier_tol': 1e-17})
opts
```

    C:\Users\user\AppData\Roaming\Python\Python37\site-packages\scipy\optimize\_hessian_update_strategy.py:187: UserWarning: delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.
      'approximations.', UserWarning)
    




     barrier_parameter: 8.388608000000012e-18
     barrier_tolerance: 8.388608000000012e-18
              cg_niter: 777
          cg_stop_cond: 2
                constr: [array([0.]), array([0.37816564, 0.22949695, 0.35766268, 0.03467473])]
           constr_nfev: [3065, 0]
           constr_nhev: [0, 0]
           constr_njev: [0, 0]
        constr_penalty: 1.0
      constr_violation: 0.0
        execution_time: 13.810054302215576
                   fun: -1.0601464032170396e-11
                  grad: array([-3.50195615e-11, -2.05084833e-11, -3.76792921e-11, -1.08951063e-11])
                   jac: [array([[1., 1., 1., 1.]]), array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])]
       lagrangian_grad: array([-1.89165944e-13,  5.77266712e-13, -4.14634055e-13,  2.65332869e-14])
               message: '`xtol` termination condition is satisfied.'
                method: 'tr_interior_point'
                  nfev: 3065
                  nhev: 0
                   nit: 456
                 niter: 456
                  njev: 0
            optimality: 5.772667115701259e-13
                status: 2
               success: True
             tr_radius: 8.23069666529829e-18
                     v: [array([3.30184431e-11]), array([ 1.81195245e-12, -1.19326931e-11,  4.24621496e-12, -2.20968035e-11])]
                     x: array([0.37816564, 0.22949695, 0.35766268, 0.03467473])




```python
optk= sco.minimize(kurtosis, eweights, method = 'trust-constr', bounds = bnds, constraints = cons, options ={'xtol': 1e-20, 'gtol': 1e-17, 'barrier_tol': 1e-13})

optk
```




     barrier_parameter: 2.0971520000000026e-16
     barrier_tolerance: 2.0971520000000026e-16
              cg_niter: 3
          cg_stop_cond: 4
                constr: [array([0.]), array([0.25, 0.25, 0.25, 0.25])]
           constr_nfev: [25, 0]
           constr_nhev: [0, 0]
           constr_njev: [0, 0]
        constr_penalty: 1.0
      constr_violation: 0.0
        execution_time: 0.36103391647338867
                   fun: 1.0443019040504316e-15
                  grad: array([5.62071765e-15, 3.29161396e-15, 6.04754542e-15, 1.74864555e-15])
                   jac: [array([[1., 1., 1., 1.]]), array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])]
       lagrangian_grad: array([ 7.68774144e-17, -4.71576932e-17,  9.96078878e-17, -1.29327609e-16])
               message: '`xtol` termination condition is satisfied.'
                method: 'tr_interior_point'
                  nfev: 25
                  nhev: 0
                   nit: 62
                 niter: 62
                  njev: 0
            optimality: 1.2932760878104553e-16
                status: 2
               success: True
             tr_radius: 3.504357843993246e-21
                     v: [array([-3.61789011e-15]), array([-1.92595012e-15,  2.79118456e-16, -2.33004743e-15,  1.73991696e-15])]
                     x: array([0.25, 0.25, 0.25, 0.25])




```python
Mstar = 0.16360798616241193
Vstar =  0.14076467102770465
Sstar = 5.809243120870251e-12
Kstar = 1.0443019040523665e-15

delta1 = Mstar - returns(weights) 
delta2 = volatility(weights) - Vstar 
delta3 = Sstar - skewness(weights) 
delta4 = kurtosis(weights) - Kstar

lambda1 = 1
lambda2 = 1
lambda3 = 1
lambda4 = 1

def objective_func(weights):
    return abs(1-returns(weights)/Mstar)**lambda1 + abs(volatility(weights)/Vstar-1)**lambda2 + abs(1-skewness(weights)/Sstar)**lambda3+ abs(kurtosis(weights)/Kstar -1)**lambda4

#objective function Z
objective_func(weights)
cons = ({'type':'eq','fun' : lambda x : np.sum(x) - 1}) #equality constraint

bnds = tuple([0,1] for x in range(num_assets))  #bonds for the parameters
```


```python
optz = sco.minimize(objective_func, eweights, method = 'trust-constr', bounds = bnds, constraints = cons, hess=None,options ={'xtol': 1e-1, 'gtol': 1e-0, 'barrier_tol': 1e-1})

optz

```




     barrier_parameter: 0.1
     barrier_tolerance: 0.1
              cg_niter: 5
          cg_stop_cond: 4
                constr: [array([0.05078471]), array([0.22561364, 0.28139576, 0.18980019, 0.35397511])]
           constr_nfev: [25, 0]
           constr_nhev: [0, 0]
           constr_njev: [0, 0]
        constr_penalty: 8.857402868469437
      constr_violation: 0.050784707618862335
        execution_time: 0.23137950897216797
                   fun: 0.7692270860341498
                  grad: array([-8.33786444, -3.94311474, -8.01176128, -2.40414252])
                   jac: [array([[1., 1., 1., 1.]]), array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])]
       lagrangian_grad: array([-0.39045012,  0.20654446, -0.31306063,  0.49696629])
               message: '`gtol` termination condition is satisfied.'
                method: 'tr_interior_point'
                  nfev: 25
                  nhev: 0
                   nit: 5
                 niter: 5
                  njev: 0
            optimality: 0.49696628606688575
                status: 1
               success: True
             tr_radius: 7.607086452981594
                     v: [array([5.42438832]), array([ 2.523026  , -1.27472913,  2.27431233, -2.52327952])]
                     x: array([0.22561364, 0.28139576, 0.18980019, 0.35397511])



# the returns



## MVSK model(PGP)


```python
opt_weights=[0.22561364, 0.28139576, 0.18980019, 0.35397511]

opt_weights = np.array(opt_weights)

returns(opt_weights) # mean
```




    0.12463462713878817




```python
np.dot(opt_weights.T,(n_returns_daily.reset_index(drop=True).T))

table_o = pd.DataFrame(np.dot(opt_weights.T,(n_returns_daily.reset_index(drop=True).T)))
table_o.hist(color='green')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001EC96725CF8>]],
          dtype=object)




![output_36_1](https://user-images.githubusercontent.com/41497195/58821550-47c40a80-8670-11e9-9985-07ef3aebeabf.png)


## MV model



```python
opt_weights=[0.25,0.25,0.25,0.25]
opt_weights = np.array(opt_weights)

returns(opt_weights)
```




    0.11770525248684999




```python
np.dot(opt_weights.T,(n_returns_daily.reset_index(drop=True).T))

table_o = pd.DataFrame(np.dot(opt_weights.T,(n_returns_daily.reset_index(drop=True).T)))
table_o.hist(color='green')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001EC96797CC0>]],
          dtype=object)




![output_39_1](https://user-images.githubusercontent.com/41497195/58821543-472b7400-8670-11e9-96fa-8c5f2c8d6b1a.png)


## Maximize skewness model




```python
opt_weights=[0.37816564, 0.22949695, 0.35766268, 0.03467473]
opt_weights = np.array(opt_weights)

returns(opt_weights)
```




    0.12137967071594552




```python
np.dot(opt_weights.T,(n_returns_daily.reset_index(drop=True).T))

table_o = pd.DataFrame(np.dot(opt_weights.T,(n_returns_daily.reset_index(drop=True).T)))
table_o.hist(color='green')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001EC9663BA20>]],
          dtype=object)




![output_42_1](https://user-images.githubusercontent.com/41497195/58821544-472b7400-8670-11e9-86f4-4e9eca8899de.png)


## Minimize variance model


```python
opt_weights=[0.32602012, 0.02572137, 0.12348349, 0.52477502]
opt_weights = np.array(opt_weights)

returns(opt_weights)
```




    0.11683232424496853




```python
np.dot(opt_weights.T,(n_returns_daily.reset_index(drop=True).T))

table_o = pd.DataFrame(np.dot(opt_weights.T,(n_returns_daily.reset_index(drop=True).T)))
table_o.hist(color='green')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001EC8EB73E48>]],
          dtype=object)



![output_45_1](https://user-images.githubusercontent.com/41497195/58840412-8a9dd680-86a0-11e9-9636-05e3b357620d.png)




```python

```
