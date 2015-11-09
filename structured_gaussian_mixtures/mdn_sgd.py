"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

INVPI = 1.0/numpy.pi
EPS = numpy.finfo(theano.config.floatX).eps

class MixtureDensityRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def random_init(self,):
        sigma_in = self.sigma_in
        mixing_in = self.mixing_in
        n_out = self.n_out
        n_components = self.n_components
        rng = self.rng

        self.W_sigma = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (sigma_in + n_out)),
                    high=numpy.sqrt(6. / (sigma_in + n_out)),
                    size=(n_out, sigma_in, n_components,)
                ),
                dtype=theano.config.floatX,
            ),
            name='W_sigma',
            borrow=True
        )
        self.W_mixing = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (mixing_in + n_out)),
                    high=numpy.sqrt(6. / (mixing_in + n_out)),
                    size=(2*mixing_in, n_components,)
                ),
                dtype=theano.config.floatX,
            ),
            name='W_mixing',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b_sigma = theano.shared(
            value=numpy.zeros(
                (1, n_out, n_components,),
                dtype=theano.config.floatX
            ),
            name='b_sigma',
            borrow=True,
            broadcastable=(True, False, False),
        )
        self.b_mixing = theano.shared(
            value=numpy.zeros(
                (1, n_components,),
                dtype=theano.config.floatX
            ),
            name='b_mixing',
            borrow=True,
            broadcastable=(True, False)
        )

    
    def __init__(self, sigma_input, mixing_input, sigma_in, mixing_in, n_out, n_components, rng, model_covars=None, model_weights=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.dim_constant = - 0.5 * T.log(2 * numpy.pi) * n_out

        self.sigma_in = sigma_in
        self.mixing_in = mixing_in
        self.n_out = n_out
        self.n_components = n_components
        self.rng = rng
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W_sigma = theano.shared(
            value=numpy.zeros((n_out, sigma_in, n_components,),
                dtype=theano.config.floatX,
            ),
            name='W_sigma',
            borrow=True
        )
        self.W_mixing = theano.shared(
            value=numpy.zeros((2*mixing_in, n_components,),
                dtype=theano.config.floatX,
            ),
            name='W_mixing',
            borrow=True
        )
        if model_covars is None:
            b_sigma = numpy.zeros((n_out, n_components))
        else:
            b_sigma = model_covars + numpy.log(1 - numpy.exp(-model_covars))
        # initialize the biases b as a vector of n_out 0s
        self.b_sigma = theano.shared(
            value=b_sigma.reshape(1, n_out, n_components).astype(
                theano.config.floatX),
            name='b_sigma',
            borrow=True,
            broadcastable=(True, False, False),
        )
        if model_weights is None:
            b_mixing = numpy.zeros(n_components)
        else:
            b_mixing = numpy.log(model_weights)
        self.b_mixing = theano.shared(
            value=b_mixing.reshape(1, n_components).astype(
                theano.config.floatX),
            name='b_mixing',
            borrow=True,
            broadcastable=(True, False)
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k

        
        sigma_input_sq = sigma_input**2
        self.invsigma_given_x = T.inv(T.maximum(T.nnet.sigmoid(theano.dot(
            sigma_input_sq, self.W_sigma) + self.b_sigma) * T.mean(sigma_input_sq, axis=1)[:, None, None], 1e-5))

        # self.invsigma_given_x = T.inv(T.maximum(T.nnet.softplus(theano.dot(
        #     sigma_input**2,
        #     self.W_sigma) + self.b_sigma), 1e-8))


        self.p_mix_given_x = T.nnet.softmax(T.dot(
            T.concatenate([mixing_input, mixing_input**2], axis=1),
            self.W_mixing) + self.b_mixing) 
        self.p_mix_given_x = T.log(self.p_mix_given_x / (T.sum(self.p_mix_given_x, axis=1)[:, None] + 10 * EPS) + EPS)

        # parameters of the model
        self.params = [self.W_sigma, self.b_sigma,
                       self.W_mixing, self.b_mixing]

        # keep track of model input
        self.sigma_input = sigma_input
        self.mixing_input = mixing_input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        log_exponent = T.sum((y**2)[:, :, None] * self.invsigma_given_x, axis=1)
        dim_constant = self.dim_constant + self.p_mix_given_x
        lpr = dim_constant + 0.5 * (
            T.sum(T.log(self.invsigma_given_x), axis=1) - log_exponent)
        max_exponent = T.max(lpr, axis=1)
        mod_exponent = lpr - max_exponent[:, None]
        gauss_mix = T.sum(T.exp(mod_exponent), axis=1)
        log_gauss = T.log(gauss_mix) + max_exponent
        return - T.mean(log_gauss)
        # end-snippet-2


def load_data():
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
    X = Z[:,:WINSIZE].astype(dtype=numpy.float32)
    Y = Z[:,WINSIZE:].astype(dtype=numpy.float32)

    X_test = Z_test[:,:WINSIZE].astype(dtype=numpy.float32)
    Y_test = Z_test[:,WINSIZE:].astype(dtype=numpy.float32)

    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(X_test, Y_test)
    valid_set_x, valid_set_y = shared_dataset(X_test, Y_test)
    train_set_x, train_set_y = shared_dataset(X, Y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_circ_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    TRAIN = "/home/mark/Projects/succotash/succotash/datasets/train_double_examples.npy"
    TEST = "/home/mark/Projects/succotash/succotash/datasets/train_double_test_examples.npy"

    # X = numpy.fft.fft(numpy.load(TRAIN))
    # X /= numpy.numpy.sqrt(X.shape[1])
    Z = numpy.load(TRAIN)
    Z_test = numpy.load(TEST)

    WINSIZE = Z.shape[1]//2
    X = numpy.abs(numpy.fft.fft(Z[:,:WINSIZE].astype(dtype=numpy.float32)))
    X /= numpy.sqrt(WINSIZE)
    Y = numpy.abs(numpy.fft.fft(Z[:,WINSIZE:].astype(dtype=numpy.float32)))
    Y /= numpy.sqrt(WINSIZE)

    X_test = numpy.abs(numpy.fft.fft(Z_test[:,:WINSIZE].astype(dtype=numpy.float32)))
    X_test /= numpy.sqrt(WINSIZE)
    Y_test = numpy.abs(numpy.fft.fft(Z_test[:,WINSIZE:].astype(dtype=numpy.float32)))
    Y_test /= numpy.sqrt(WINSIZE)

    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(X_test, Y_test)
    valid_set_x, valid_set_y = shared_dataset(X_test, Y_test)
    train_set_x, train_set_y = shared_dataset(X, Y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


