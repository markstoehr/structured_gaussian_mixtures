%autoindent
import numpy
import theano
from theano import tensor

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)


num_timesteps = 10
num_sequences = 3
num_dim = 2
num_components = 3

x_n = (numpy.arange(num_timesteps * num_sequences * num_dim,
                               dtype=theano.config.floatX)
                  .reshape(num_sequences, num_timesteps, num_dim)
                  .swapaxes(0, 1))
y_n = (numpy.arange(num_timesteps * num_sequences,
                               dtype=theano.config.floatX)
                  .reshape(num_sequences, num_timesteps)
                  .T + 2)
x = tensor.tensor3('x', dtype=theano.config.floatX)
y = tensor.matrix('y', dtype=theano.config.floatX)
W_n_sigma = numpy.random.uniform(
    low=-1,
    high=1,
    size=(num_dim,))
W_sigma = theano.shared(W_n_sigma, borrow=True, name='W_sigma')

W_n_mu = numpy.random.uniform(
    low=-1,
    high=1,
    size=(num_dim,))
W_mu = theano.shared(W_n_mu, borrow=True, name='W_mu')

W_n_mix = numpy.random.uniform(
    low=-1,
    high=1,
    size=(num_dim, num_components,))
W_mix = theano.shared(W_n_mix, borrow=True, name='W_mix')


# check whether scan does what I think it does
def step(x_, y_, ll_):
    v = tensor.mean((x_[:, 1:] - x_[:, :-1])**2, axis=-1)
    mu = tensor.dot(x_, W_mu)
    invsigma = tensor.maximum(tensor.nnet.sigmoid(
        tensor.dot(x_, W_sigma)), 1e-8) / v
    return (mu - y_)**2 * invsigma


lls, updates = theano.scan(step, sequences=[x, y],
                           outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                      num_sequences)],
                           name='lls',
                           n_steps=num_timesteps)

f_lls = theano.function([x, y], lls)
f_updates = theano.function([], updates)

def sigmoid(z):
    less_than_mask = z < -30
    greater_than_mask = z > 30
    in_range_mask = (- less_than_mask) * (- greater_than_mask)
    out = numpy.empty(z.shape, dtype=float)
    out[in_range_mask] = 1.0/(1+numpy.exp(-z[in_range_mask]))
    out[less_than_mask] = 0.0
    out[greater_than_mask] = 1.0
    return out

mu_n = numpy.dot(x_n, W_n_mu)
invsigma_n = numpy.maximum(sigmoid(numpy.dot(x_n, W_n_sigma)), 1e-8)
lls_n = (mu_n - y_n)**2 * invsigma_n

