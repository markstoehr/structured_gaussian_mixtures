from __future__ import print_function, division
import cPickle
import gzip
import os
import sys
import timeit
import numpy
import theano
from theano import tensor

import mdn_one_ahead

# parameters
batch_size = 100
L1_reg=0.00
L2_reg=0.0001
n_epochs=100

sigma_in = 320
mixing_in = 320
n_components = 3

EPS = numpy.finfo(theano.config.floatX).eps

# load data
datasets = mdn_one_ahead.load_data()

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

X = train_set_x.get_value(borrow=True)[:20].copy()
Y = train_set_y.get_value(borrow=True)[:20].copy()

n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

print( '... building the model')

# allocate symbolic variables for the data
index = tensor.lscalar()  # index to a [mini]batch
x = tensor.matrix('x')  # the data is presented as rasterized images
y = tensor.vector('y')  # the labels are presented as 1D vector of

# model parameters
W_sigma = theano.shared(
    value=numpy.zeros((sigma_in, n_components,),
        dtype=theano.config.floatX,
    ),
    name='W_sigma',
    borrow=True
)
W_mu = theano.shared(
    value=numpy.zeros((sigma_in, n_components,),
        dtype=theano.config.floatX,
    ),
    name='W_mu',
    borrow=True
)
W_mixing = theano.shared(
    value=numpy.zeros((mixing_in, n_components,),
        dtype=theano.config.floatX,
    ),
    name='W_mixing',
    borrow=True
)

b_sigma = theano.shared(
    value=numpy.zeros(n_components,
        dtype=theano.config.floatX,
    ),
    name='b_sigma',
    borrow=True
)
b_mu = theano.shared(
    value=numpy.zeros(n_components,
        dtype=theano.config.floatX,
    ),
    name='b_mu',
    borrow=True
)
b_mixing = theano.shared(
    value=numpy.zeros(n_components,
        dtype=theano.config.floatX,
    ),
    name='b_mixing',
    borrow=True
)

x_sq = (x[:,1:] - x[:,:-1])**2
invsigma_given_x = tensor.inv(tensor.maximum(
    tensor.exp(theano.dot(
        x, W_sigma) + b_sigma)
    * tensor.mean(x_sq, axis=1)[:, None], 1e-3))

mu = theano.dot(x, W_mu) + b_mu

p_mix_given_x = tensor.nnet.softmax(tensor.dot(x, W_mixing) + b_mixing) 
p_mix_given_x = tensor.log(p_mix_given_x / (
    tensor.sum(p_mix_given_x, axis=1)[:, None] + 10 * EPS) + EPS)

log_exponent = (y[:, None] - mu)**2 * invsigma_given_x
dim_constant = 0.5*numpy.log(2*numpy.pi) + p_mix_given_x
lpr = dim_constant + 0.5 * (
    tensor.log(invsigma_given_x) - log_exponent)

f = theano.function(inputs=[x, y], outputs=lpr)

max_exponent = tensor.max(lpr, axis=1)
mod_exponent = lpr - max_exponent[:, None]
gauss_mix = tensor.sum(tensor.exp(mod_exponent), axis=1)
log_gauss = tensor.log(gauss_mix) + max_exponent

f = theano.function(inputs=[x, y], outputs=log_gauss)

