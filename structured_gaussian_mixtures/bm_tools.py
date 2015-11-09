from __future__ import division
import numpy as np

# helper functions to make
# the computations easier
class OnlineLogsumexp:
    def __init__(self):
        self.total = 0.0
        self.compensation = 0.0
        self.n = 0
        self.max_val = -np.inf
    def add(self,x):
        if x > self.max_val:
            self.total *= np.exp(self.max_val - x)
            self.max_val = x
        self.n += 1
        a = ( np.exp(x - self.max_val) - self.total)/self.n
        y = a - self.compensation
        t = self.total + y
        self.compensation = (t-self.total) - y
        self.total = t
    def out(self):
        return np.log(self.total) + np.log(self.n) + self.max_val

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))

def log1pexp(x):
    out = x.copy()
    idx = x < 50
    out[idx] = np.log(1+np.exp(x[idx]))
    return out

def logsumexp(x):
    if x.ndim > 1:
        max_x = x.max(axis=1)
        return np.log(np.exp(x-max_x[:,np.newaxis]+4).sum(axis=1)) + max_x -4
    else:
        max_x = x.max()
        return np.log(np.exp(x-max_x +4).sum()) + max_x -4
