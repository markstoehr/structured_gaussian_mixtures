from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import mdn_lstm
import imdb

%autoindent

SEED=1234
dim_proj=128  # word embeding dimension and LSTM number of hidden units.
patience=10  # Number of epoch to wait before early stop if no progress
max_epochs=5000  # The maximum number of epoch to run
dispFreq=10  # Display to stdout the training progress every N updates
decay_c=0.  # Weight decay for the classifier applied to the U weights.
lrate=0.0001  # Learning rate for sgd (not used for adadelta and rmsprop)
n_words=10000  # Vocabulary size
optimizer=mdn_lstm.adadelta  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
encoder='lstm'  # TODO: can be removed must be lstm.
saveto='lstm_model.npz'
validFreq=370  # Compute the validation error after this number of update.
saveFreq=1110  # Save the parameters after every saveFreq updates
maxlen=100  # Sequence longer then this get ignored
batch_size=16  # The batch size during training.
valid_batch_size=64  # The batch size used for validation/test set.
dataset='imdb'

# Parameter for extra option
noise_std=0.
use_dropout=True  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
reload_model=None  # Path to a saved model we want to start from.
test_size=-1  # If >0, we keep only this number of test example.

# Model options
model_options = locals().copy()
# print( "model options", model_options)
    
load_data, prepare_data = imdb.load_data, imdb.prepare_data
# 

print( 'Loading data')
X, Y, X_test, Y_test = mdn_lstm.load_data(predict=dim_proj)

n_components = 3

model_options['n_components'] = n_components

print( 'Building model')
# This create the initial parameters as numpy ndarrays.
# Dict name (string) -> numpy ndarray


# params = init_params(model_options)
# unfoled into a thing
params = OrderedDict()
# embedding
randn = numpy.random.rand(model_options['n_words'],
                          model_options['dim_proj'])

params = mdn_lstm.param_init_lstm(model_options,
                                  params,
                                  prefix=model_options['encoder'])
# params = mdn_lstm.param_init_rnn(model_options,
#                                  params,
#                                  prefix='rnn')
# classifier
params['gmm_U_mu'] = 0.01 * numpy.random.randn(model_options['dim_proj'],
                                        model_options['n_components']).astype(config.floatX)
params['gmm_b_mu'] = numpy.zeros((model_options['n_components'],)).astype(config.floatX)
params['gmm_U_mix'] = 0.01 * numpy.random.randn(model_options['dim_proj'],
                                        model_options['n_components']).astype(config.floatX)
params['gmm_b_mix'] = numpy.zeros((model_options['n_components'],)).astype(config.floatX)
params['gmm_U_sigma'] = 0.01 * numpy.random.randn(model_options['dim_proj'],
                                        model_options['n_components']).astype(config.floatX)
params['gmm_b_sigma'] = numpy.zeros((model_options['n_components'],)).astype(config.floatX)


# This create Theano Shared Variable from the parameters.
# Dict name (string) -> Theano Tensor Shared Variable
# params and tparams have different copy of the weights.
# tparams = init_tparams(params)

tparams = OrderedDict()
for kk, pp in params.iteritems():
    tparams[kk] = theano.shared(params[kk], name=kk)

trng = RandomStreams(SEED)

# Used for dropout.
use_noise = theano.shared(mdn_lstm.numpy_floatX(0.))

x = tensor.tensor3('x', dtype=config.floatX)
mask = tensor.matrix('mask', dtype=config.floatX)
y = tensor.matrix('y', dtype=config.floatX)


proj = mdn_lstm.lstm_layer_all(tparams, x, model_options,
                           prefix=model_options['encoder'],
                           mask=mask)


nll = mdn_lstm.gauss_negativeloglikelihood(x, proj, y, tparams, prefix='gmm')
yhat = mdn_lstm.gauss_means(x, proj, tparams, prefix='gmm')

f_squared_error = theano.function([x, mask, y], tensor.mean(0.5*(yhat - y)**2))

f_mu = theano.function([x, mask], yhat)

f_nll = theano.function([x, mask, y], nll)

#f_var_estimate = theano.function([x], var_estimate)

#nll = mdn_lstm.gauss_negativeloglikelihood(x, proj, y, tparams, prefix='gmm')

f_proj = theano.function([x, mask], proj.swapaxes(0, 1))

# f_nll = theano.function([x, mask, y], nll)





#if model_options['use_dropout']:
#    proj = mdn_lstm.dropout_layer(proj, use_noise, trng)

f_cost = theano.function([x, mask, y], nll, name='f_cost')


grads = tensor.grad(nll, wrt=tparams.values())
f_grad = theano.function([x, mask, y], grads, name='f_grad')

lr = tensor.scalar(name='lr')
f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                    x, mask, y, nll)

kf_valid = mdn_lstm.get_minibatches_idx(X_test.shape[0], valid_batch_size)
kf_test = mdn_lstm.get_minibatches_idx(X_test.shape[0], valid_batch_size)

print( "%d train examples" % len(X))
print( "%d valid examples" % len(X_test))
print( "%d test examples" % len(X_test))

history_errs = []
best_p = None
bad_count = 0

if validFreq == -1:
    validFreq = len(train[0]) / batch_size
if saveFreq == -1:
    saveFreq = len(train[0]) / batch_size

uidx = 0  # the number of update done
estop = False  # early stop
start_time = time.time()
try:
    for eidx in xrange(max_epochs):
        n_samples = 0
        # Get new shuffled index for the training set.
        kf = mdn_lstm.get_minibatches_idx(len(X), batch_size, shuffle=True)
        for _, train_index in kf:
            uidx += 1
            use_noise.set_value(1.)
            # Select the random examples for this minibatch
            x = mdn_lstm.prepare_data(X[train_index], predict=dim_proj)
            y = Y[train_index]
            mask = numpy.ones(x.shape[:2], dtype=theano.config.floatX)
            n_samples += x.shape[1]
            cost = f_grad_shared(x, mask, y)
            f_update(lrate)
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                out = 1., 1., 1.
                break
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
            if saveto and numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                print 'Done'
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_err = mdn_lstm.pred_error(f_nll, mdn_lstm.prepare_data, X_test, Y_test, dim_proj,
                                       kf_valid)
                test_err = valid_err
                history_errs.append([valid_err, test_err])
                if (uidx == 0 or
                    valid_err <= numpy.array(history_errs)[:,
                                                           0].min()):
                    best_p = unzip(tparams)
                    bad_counter = 0
                print ('Train ', train_err, 'Valid ', valid_err,
                       'Test ', test_err)
                if (len(history_errs) > patience and
                    valid_err >= numpy.array(history_errs)[:-patience,
                                                           0].min()):
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break
        print 'Seen %d samples' % n_samples
        if estop:
            break
except KeyboardInterrupt:
    print "Training interupted"

