"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from mdn_sgd import MixtureDensityRegression, load_data

INVPI = 1.0/numpy.pi
EPS = numpy.finfo(theano.config.floatX).eps

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
    Y = Z[:,WINSIZE].astype(dtype=numpy.float32)

    X_test = Z_test[:,:WINSIZE].astype(dtype=numpy.float32)
    Y_test = Z_test[:,WINSIZE].astype(dtype=numpy.float32)

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
        n_out = 1
        n_components = self.n_components
        rng = self.rng

        self.W_sigma = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (sigma_in + n_out)),
                    high=numpy.sqrt(6. / (sigma_in + n_out)),
                    size=(sigma_in, n_components,)
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
                (1, n_components,),
                dtype=theano.config.floatX
            ),
            name='b_sigma',
            borrow=True,
            broadcastable=(True, False),
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

    
    def __init__(self, sigma_input, mixing_input, sigma_in, mixing_in, n_components, rng, model_covars=None, model_weights=None):
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
        self.dim_constant = - 0.5 * T.log(2 * numpy.pi)

        self.sigma_in = sigma_in
        self.mixing_in = mixing_in
        self.n_out = 1
        self.n_components = n_components
        self.rng = rng
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W_sigma = theano.shared(
            value=numpy.zeros((sigma_in, n_components,),
                dtype=theano.config.floatX,
            ),
            name='W_sigma',
            borrow=True
        )
        self.W_mu = theano.shared(
            value=numpy.zeros((sigma_in, n_components,),
                dtype=theano.config.floatX,
            ),
            name='W_mu',
            borrow=True
        )
        self.W_mixing = theano.shared(
            value=numpy.zeros((mixing_in, n_components,),
                dtype=theano.config.floatX,
            ),
            name='W_mixing',
            borrow=True
        )
        if model_covars is None:
            b_sigma = numpy.zeros(n_components)
        else:
            b_sigma = model_covars + numpy.log(1 - numpy.exp(-model_covars))
        # initialize the biases b as a vector of n_out 0s
        self.b_sigma = theano.shared(
            value=b_sigma.reshape(n_components).astype(
                theano.config.floatX),
            name='b_sigma',
            borrow=True,
        )
        self.b_mu = theano.shared(
            value=numpy.zeros(n_components).astype(
                theano.config.floatX),
            name='b_sigma',
            borrow=True,
        )
        if model_weights is None:
            b_mixing = numpy.zeros(n_components)
        else:
            b_mixing = numpy.log(model_weights)
        self.b_mixing = theano.shared(
            value=b_mixing.reshape(n_components).astype(
                theano.config.floatX),
            name='b_mixing',
            borrow=True,
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k

        
        sigma_input_sq = (sigma_input[:,1:] - sigma_input[:,:-1])**2
        self.invsigma_given_x = T.maximum(T.nnet.sigmoid(theano.dot(
                sigma_input, self.W_sigma) + self.b_sigma)
                                          , 1e-8)/ T.mean(sigma_input_sq, axis=1)[:, None]

        self.mu = theano.dot(sigma_input, self.W_mu) + self.b_mu
        
        # self.invsigma_given_x = T.inv(T.maximum(T.nnet.softplus(theano.dot(
        #     sigma_input**2,
        #     self.W_sigma) + self.b_sigma), 1e-8))


        self.p_mix_given_x = T.maximum(T.minimum(T.nnet.softmax(T.dot(
            mixing_input, self.W_mixing) + self.b_mixing), 1e-6), 1-1e-6)
        self.p_mix_given_x = T.log(self.p_mix_given_x / (T.sum(self.p_mix_given_x, axis=1)[:, None] + 10 * EPS) + EPS)

        # parameters of the model
        self.params = [self.W_sigma, self.b_sigma, self.W_mu, self.b_mu,
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
        log_exponent = (y[:, None] - self.mu)**2 * self.invsigma_given_x
        dim_constant = self.dim_constant + self.p_mix_given_x
        lpr = dim_constant + 0.5 * (
            T.log(self.invsigma_given_x) - log_exponent)
        max_exponent = T.max(lpr, axis=1)
        mod_exponent = lpr - max_exponent[:, None]
        gauss_mix = T.sum(T.exp(mod_exponent), axis=1)
        log_gauss = T.log(gauss_mix) + max_exponent
        return - T.mean(log_gauss)
        # end-snippet-2


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.n_in = n_in
        self.n_out = n_out
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hiddens):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        if type(n_hiddens) == list:
            self.n_layers = len(n_hiddens)
            self.n_hiddens = n_hiddens
        else:
            self.n_layers = 1
            self.n_hiddens = [n_hiddens]

        self.hiddenLayers = [HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=self.n_hiddens[0],
            activation=T.nnet.relu
        )]
        for l in range(1, self.n_layers):
            self.hiddenLayers.append(HiddenLayer(
            rng=rng,
            input=self.hiddenLayers[-1].output,
            n_in=self.hiddenLayers[-1].n_out,
            n_out=self.n_hiddens[l],
            activation=T.nnet.relu
        ))

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.mixRegressionLayer = MixtureDensityRegression(
            sigma_input=self.hiddenLayers[-1].output,
            mixing_input=self.hiddenLayers[-1].output,
            sigma_in=self.hiddenLayers[-1].n_out,
            mixing_in=self.hiddenLayers[-1].n_out,
            n_components=3,
            rng=rng,
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
              abs(self.mixRegressionLayer.W_sigma).sum()
            + abs(self.mixRegressionLayer.W_mu).sum()
            + abs(self.mixRegressionLayer.W_mixing ** 2).sum()
        )
        for hl in self.hiddenLayers:
            self.L1 += abs(hl.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.mixRegressionLayer.W_sigma ** 2).sum()
            + (self.mixRegressionLayer.W_mu ** 2).sum()
            + (self.mixRegressionLayer.W_mixing ** 2).sum()
        )
        for hl in self.hiddenLayers:
            self.L2_sqr += (hl.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.mixRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.mixRegressionLayer.negative_log_likelihood

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.mixRegressionLayer.params
        for hl in self.hiddenLayers:
            self.params.extend(hl.params)

        # end-snippet-3

        # keep track of model input
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.vector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
