
import numpy as np
from scipy.stats import binom
from find_correlation import correlation
from IPython import embed

# Find k for binomial distribution that corresponds to cumulative probability 
# approximately equal to 0.001 = 0.1%
print(binom.cdf(k=49511, n=100000, p=0.5))
# 0.0010022415200593084

n = 100000
k = 49511

col1 = np.zeros(n)
col2 = np.zeros(n) 
# I want the mean to be close to 0.5, and columns equal in n - k inputs. 

for i in range(int(n/2)):
    col1[i] = 1
    col2[2*i] = 1

for i in range(int(n/2 - k)):
    col1[2*i] = 0

diff = abs(col1-col2)
print(np.mean(diff))
# 0.50489
print(correlation(col1, col2))
# -0.009780467754231225
# embed()
