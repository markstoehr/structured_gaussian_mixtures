from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

NORMLOGCONSTANT = 0.5 * numpy.log(2*numpy.pi)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def load_data(predict=80):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    TRAIN = "/home/mark/Projects/succotash/succotash/datasets/train_double_examples.npy"
    TEST = "/home/mark/Projects/succotash/succotash/datasets/train_double_test_examples.npy"

    # X = numpy.fft.fft(numpy.load(TRAIN))
    # X /= numpy.sqrt(X.shape[1])
    Z = numpy.load(TRAIN)
    Z_test = numpy.load(TEST)

    WINSIZE = Z.shape[1]//2
    X = Z[:,:WINSIZE+predict].astype(dtype=numpy.float64)
    Y = Z[:,predict:WINSIZE+predict].astype(dtype=numpy.float64)

    X_test = Z_test[:,:WINSIZE+predict].astype(dtype=numpy.float64)
    Y_test = Z_test[:,predict:WINSIZE+predict].astype(dtype=numpy.float64)

    return X, Y, X_test, Y_test

def prepare_data(x, WINSIZE=320, predict=80):
    """Create sequences"""
    n_sequences, n_timesteps = x.shape
    x = numpy.lib.stride_tricks.as_strided(
        x, shape=(n_sequences, WINSIZE, predict),
        strides=(x.strides[0], x.strides[1], x.strides[1])
    ).copy().swapaxes(0, 1)
    return x


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost

def pred_error(f_squared_error, prepare_data, X, Y, dim_proj, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x = prepare_data(X[valid_index], predict=dim_proj)
        y = Y[valid_index]
        mask = numpy.ones(x.shape[:2], dtype=theano.config.floatX)
        valid_err += f_squared_error(x, mask, y)
    valid_err = 1. - numpy_floatX(valid_err) / len(X)

    return valid_err


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def hidden_layer(options, n_in, n_out, x, prefix='hidden'):
    U = numpy.random.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ).astype(theano.config.floatX)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((n_out,), dtype=theano.config.floatX)
    params[_p(prefix, 'b')] = b
    return params    

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

def gauss_means(x, h, tparams, prefix='gmm'):
    h = h.swapaxes(0,1)
    num_sequences = h.shape[0]
    num_timesteps = h.shape[1]
    proj_dim = h.shape[2]
    h = h.reshape((num_sequences * num_timesteps, proj_dim))
    mix = tensor.nnet.softmax(tensor.dot(h, tparams[_p(prefix, 'U_mix')]) + tparams[_p(prefix, 'b_mix')])
    mu = tensor.dot(h, tparams[_p(prefix, 'U_mu')]) + tparams[_p(prefix, 'b_mu')] + x[:, :, -1].T.flatten()[:, None]
    return tensor.sum(mix * mu, axis=-1).reshape((num_sequences, num_timesteps))
    
    
def gauss_negativeloglikelihood(x, h, y, tparams, prefix='gmm'):
    h = h.swapaxes(0,1)
    num_sequences = h.shape[0]
    num_timesteps = h.shape[1]
    proj_dim = h.shape[2]
    h = h.reshape((num_sequences * num_timesteps, proj_dim))
    y = y.flatten()
    var_estimate = tensor.mean((x[:, :, 1:] - x[:, :, :-1])**2, axis=-1).T.flatten()
    mix = tensor.nnet.softmax(tensor.dot(h, tparams[_p(prefix, 'U_mix')]) + tparams[_p(prefix, 'b_mix')])
    mu = tensor.dot(h, tparams[_p(prefix, 'U_mu')]) + tparams[_p(prefix, 'b_mu')] + x[:, :, -1].T.flatten()[:, None]
    invsigma = (tensor.maximum(
        tensor.nnet.sigmoid(
            tensor.dot(h, tparams[_p(prefix, 'U_sigma')]) + tparams[_p(prefix, 'b_sigma')]), 1e-8)
                     / var_estimate[:, None])
    log_exponent = (y[:, None] - mu)**2 * invsigma
    dim_constant = NORMLOGCONSTANT + tensor.log(mix)
    lpr = dim_constant + 0.5 * (tensor.log(invsigma) - log_exponent)
    max_exponent = tensor.max(lpr, axis=-1)
    mod_exponent = lpr - max_exponent[:, None]
    gauss_mix = tensor.sum(tensor.exp(mod_exponent), axis=-1)
    return -tensor.mean(tensor.log(gauss_mix) + max_exponent)
    
                     

def lstm_layer_all(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

def param_init_rnn(options, params, prefix='rnn'):
    W = ortho_weight(options['dim_proj'])
    params[_p(prefix, 'W')] = W    
    U = ortho_weight(options['dim_proj'])
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    return params

def rnn_layer_all(tparams, state_below, options, prefix='rnn'):
    nsteps = state_below.shape[0]
    n_samples = state_below.shape[1]

    def _step(x_, h_):
        h = tensor.dot(h_, tparams[_p(prefix, 'U')])
        h += x_
        return h

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = param_init_lstm(options,
                             params,
                             prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params
