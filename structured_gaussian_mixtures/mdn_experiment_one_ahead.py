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
n_epochs=200
learning_rate = 0.001
momentum = 0.9

sigma_in = 320
mixing_in = 320
n_components = 5

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

rng = numpy.random.RandomState(1234)

classifier = mdn_one_ahead.MLP(
        rng=rng,
        input=x,
        n_in=320,
        n_hiddens=[300, 300, 300, 300]
    )


cost = (
        classifier.negative_log_likelihood(y)
        + L2_reg * classifier.L2_sqr
    )


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

gparams = [tensor.grad(cost, param) for param in classifier.params]

updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
]

model_gradients = theano.function(
    inputs = [x, y], outputs=gparams)

train_gradients = theano.function(
    inputs=[index],
    outputs=gparams,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

print('... training')

# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
# found
improvement_threshold = 0.99995  # a relative improvement of this much is
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
        gs = train_gradients(minibatch_index)
        if any(numpy.any(numpy.isnan(g)) for g in gs):
            import pdb; pdb.set_trace()
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

# l = 7.752, tanh, 3 components, 20 hid, 1 hidlayer,
# l = 5.057, relu, 3 components, (100, 100) hid
# l = 4.865, relu, 5 components, (150, 150, 150) hid
