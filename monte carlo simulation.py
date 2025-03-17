import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

pHeads = 0.5
N = 100000
M = 3
CoinFlips = np.random.binomial(1, pHeads, N * M).reshape([N, M])
results = CoinFlips.prod(axis=1)

print('Exact answer {:.4f}:'.format(pHeads ** M))
print(50 * '-')

print('Probability of 3 heads in a row is = [{:.4f}]'.format(results.mean()))
print('Standard error = [{:.4f}]'.format(np.std(results) / np.sqrt(N)))

# no need for loop here because of numpy


# asian options has no formula (only for geometric and not arithmetic average)
# so we use Monte Carlo simulation
# As you increase the number of simulations, the standard error decreases, 
# the error is inversely proportional to the square root of the number of simulations

# boostrapping (assumes future returns will be similar to past returns):

# or you can make a forward looking distribution of the stock price and 
# generate random numbers probabilistically based on that distribution
# probabilistic modes: 
# - gaussian (normally distributed)
# pros: just need mean and standard deviation


# in the stock market, monte carlo works by looking at the past behavior of a stock
# and then simulating many random walks of the stock price
# the more simulations you run, the more accurate the model becomes


# make an assunption of distrubiton, apply mechanism, discount it to present value
# then take average of all the prices

# monte carlo (european example):
# this is monte carlo for european so its diderent, its focused on terminal value. 
# We have to get price path and get average of prices and then commpute if max of 0 to determine price and discount it 
# to present value

import numpy as np
from scipy.stats import norm

# Define the Black-Scholes-Merton (BSM) model function
def BSM(S, K, r, T, sigma, option):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == 'call':
        C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return C
    elif option == 'put':
        P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return P

# Parameters
risk_free_rate = 0.05
volatility = 0.2
T = 30 / 252
Sigma = 0.2
t = 0.0
S = 100.0
K = np.array([105.0, 100.0, 95.0])  # Corrected variable name to K

N = 10000  # Increased number of simulations for better accuracy

# Drift and shock calculations
drift = (risk_free_rate - 0.5 * volatility ** 2) * (T - t)
shock = Sigma * np.sqrt(T - t)

# Monte Carlo simulation
terminalvalues = S * np.exp(drift + shock * np.random.normal(0, 1, N))
payoffcall = terminalvalues.reshape((-1, 1)) * np.ones((N, len(K))) - K
payoffput = K - terminalvalues.reshape((-1, 1)) * np.ones((N, len(K)))

C = np.fmax(0.0, payoffcall) * np.exp(-risk_free_rate * (T - t))
P = np.fmax(0.0, payoffput) * np.exp(-risk_free_rate * (T - t))

price_Call = C.mean(axis=0)
price_Put = P.mean(axis=0)

SE_C = C.std(axis=0) / np.sqrt(N)
SE_P = P.std(axis=0) / np.sqrt(N)

# Print results
for i in range(len(K)):
    print('Exact Call price for K = {:.2f} is {:.4f}'.format(K[i], BSM(S, K[i], risk_free_rate, T, volatility, 'call')))
    print('MC Call price for K = {:.2f} is {:.4f}'.format(K[i], price_Call[i]))
    print('Standard Error for Call price for K = {:.2f} is {:.4f}'.format(K[i], SE_C[i]))
    print(50 * '-')
    print('Exact Put price for K = {:.2f} is {:.4f}'.format(K[i], BSM(S, K[i], risk_free_rate, T, volatility, 'put')))
    print('MC Put price for K = {:.2f} is {:.4f}'.format(K[i], price_Put[i]))
    print('Standard Error for Put price for K = {:.2f} is {:.4f}'.format(K[i], SE_P[i]))
    print(50 * '-')