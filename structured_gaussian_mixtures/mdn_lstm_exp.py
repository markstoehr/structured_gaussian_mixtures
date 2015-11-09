
import mdn_lstm
reload(mdn_lstm)
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)


dim_proj=128  # word embeding dimension and LSTM number of hidden units.
patience=10  # Number of epoch to wait before early stop if no progress
max_epochs=5000  # The maximum number of epoch to run
dispFreq=10  # Display to stdout the training progress every N updates
decay_c=0.  # Weight decay for the classifier applied to the U weights.
lrate=0.0001  # Learning rate for sgd (not used for adadelta and rmsprop)
n_components=5
optimizer=mdn_lstm.adadelta  # sgd adadelta and rmsprop available sgd very hard to use not recommanded (probably need momentum and decaying learning rate).
encoder='lstm'  # TODO: can be removed must be lstm.
saveto='lstm_model.npz'  # The best model will be saved there
validFreq=370  # Compute the validation error after this number of update.
saveFreq=1110  # Save the parameters after every saveFreq updates
maxlen=100  # Sequence longer then this get ignored
batch_size=16  # The batch size during training.
valid_batch_size=64  # The batch size used for validation/test set.
dataset='imdb'

noise_std=0.
use_dropout=True  # if False slightly faster but worst test error
                   # This frequently need a bigger model.
reload_model=None  # Path to a saved model we want to start from.
test_size=-1  # If >0 we keep only this number of test example.

model_options = locals().copy()
# print "model options", model_options


X, Y, X_test, Y_test = mdn_lstm.load_data()


params = mdn_lstm.init_params(model_options)
tparams = mdn_lstm.init_tparams(params)
(use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = mdn_lstm.build_model(tparams, model_options)

f_cost = theano.function([x, mask, y], cost, name='f_cost')
grads = tensor.grad(cost, wrt=tparams.values())
f_grad = theano.function([x, mask, y], grads, name='f_grad')

lr = tensor.scalar(name='lr')
f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                    x, mask, y, cost)

print 'Optimization'

kf_test = get_minibatches_idx(len(X_test[0]), valid_batch_size)
