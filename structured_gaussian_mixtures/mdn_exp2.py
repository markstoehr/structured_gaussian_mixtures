from __future__ import print_function
import cPickle
import gzip
import os
import sys
import timeit
import numpy
import theano
from theano import tensor

# import mdn_sgd; reload(mdn_sgd)
from mdn_sgd import MixtureDensityRegression, load_circ_data

batch_size = 300
learning_rate = 0.000001
momentum = 0.1

EPS = numpy.finfo(theano.config.floatX).eps
datasets = load_circ_data()

WINSIZE = datasets[0][0].get_value().shape[1]

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

X = train_set_y.get_value(borrow=True)[:20].copy()
X_test = test_set_y.get_value(borrow=True).copy()

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

print('... building the model')

index = tensor.lscalar()  # index to a [mini]batch
x = tensor.matrix('x')  # the data is presented as rasterized images
y = tensor.matrix('y')  # the labels are presented as 1D vector of

rng = numpy.random.RandomState(1234)

n_components = 5
model_covars = numpy.ones((n_components, WINSIZE))
model_weights = numpy.ones(n_components, dtype=theano.config.floatX)/n_components
model_covars_trans = model_covars + numpy.log(1 - numpy.exp(-model_covars))

mc = theano.shared(
    value=model_covars_trans.T.copy().reshape(1, WINSIZE, n_components,).astype(
        theano.config.floatX),
    name='mc',
    borrow=True,
    broadcastable=(True, False, False),)
mw = theano.shared(
    value=numpy.log(model_weights.copy().astype(
        theano.config.floatX)),
    name='mw',
    borrow=True,)

Wc = theano.shared(
    value=numpy.zeros((n_out, sigma_in, n_components,), dtype=theano.config.floatX),
    name='Wc',
    borrow=True,)

invsigma_given_x = tensor.inv(tensor.maximum(tensor.nnet.softplus(theano.dot(x, Wc) + mc), 1e-8))
f = theano.function(
        inputs=[x,],
        outputs=invsigma_given_x,
    )
p_mix_given_x = tensor.nnet.softmax(mw) 
p_mix_given_x = tensor.log(p_mix_given_x / (tensor.sum(p_mix_given_x, axis=1)[:, None] + 10 * EPS) + EPS)
log_exponent = tensor.sum((y**2)[:, :, None] * invsigma_given_x, axis=1)
f = theano.function(
        inputs=[x, y],
        outputs=log_exponent,
    )

dim_constant = - 0.5 * WINSIZE * tensor.log(2 * numpy.pi) + p_mix_given_x
lpr = dim_constant + 0.5 * (
            tensor.sum(tensor.log(invsigma_given_x), axis=1) - log_exponent)
f = theano.function(inputs=[x, y], outputs=lpr)
max_exponent = tensor.max(lpr, axis=1)
mod_exponent = lpr - max_exponent[:, None]
gauss_mix = tensor.sum(tensor.exp(mod_exponent), axis=1)
log_gauss = tensor.log(gauss_mix) + max_exponent
res = - tensor.mean(log_gauss)

f = theano.function(
        inputs=[x, y],
        outputs=res,
    )


df = theano.function(
        inputs=[x],
        outputs=tensor.grad(tensor.sum(p_mix_given_x), wrt=W_mixing),
    )

params = [W_sigma, b_sigma,
          W_mixing, b_mixing]



mdn_model = MixtureDensityRegression(x, x, WINSIZE, WINSIZE, WINSIZE, n_components, rng, model_weights=model_weights)

mdn_nll = mdn_model.negative_log_likelihood(y)

f = theano.function(inputs=[x, y], outputs=mdn_nll)
        # self.invsigma_given_x = T.inv(T.sigmoid(theano.dot(
        #     sigma_input**2,
        #     self.W_sigma) + self.b_sigma), 1e-8))

X = train_set_x.get_value(borrow=True)[:20]

sigma_in = WINSIZE
n_out = WINSIZE
mixing_in = WINSIZE
W_sigma = theano.shared(
            value=(numpy.random.rand(
                 n_out, sigma_in, n_components,
            ).astype(theano.config.floatX) - 0.5)* 0.001,
            name='W_sigma',
            borrow=True
        )
W_mixing = theano.shared(
            value=numpy.random.randn(
                mixing_in, n_components,
            ).astype(theano.config.floatX),
            name='W_mixing',
            borrow=True
        )
b_sigma = theano.shared(
            value=numpy.random.randn(
                1, n_out, n_components,
            ).astype(theano.config.floatX) * 0.0001,
            name='b_sigma',
            borrow=True,
    broadcastable=(True, False, False)
        )
# b_sigma = theano.shared(
#             value=numpy.zeros(
#                 (n_out, n_components),
#                 dtype=theano.config.floatX
#             ),
#             name='b_sigma',
#             borrow=True
#         )
b_mixing = theano.shared(
            value=numpy.zeros(
                (1, n_components,),
                dtype=theano.config.floatX
            ),
            name='b_mixing',
            borrow=True,
    broadcastable=(True, False)
        )
invsigma_given_x = tensor.inv(tensor.nnet.softplus(tensor.dot(x, W_sigma) + b_sigma))
p_mix_given_x = tensor.nnet.softmax(tensor.dot(x, W_mixing) + b_mixing) 
p_mix_given_x = tensor.log(p_mix_given_x / (tensor.sum(p_mix_given_x, axis=1)[:, None] + 10 * EPS) + EPS)

f = theano.function(
        inputs=[x],
        outputs=p_mix_given_x,
    )
df = theano.function(
        inputs=[x],
        outputs=tensor.grad(tensor.sum(p_mix_given_x), wrt=W_mixing),
    )

params = [W_sigma, b_sigma,
          W_mixing, b_mixing]
log_exponent = - 0.5 * tensor.sum((y**2)[:, :, None] * invsigma_given_x, axis=1)
dim_constant = - 0.5 * tensor.log(2 * numpy.pi)
lpr = (dim_constant[None, None] + p_mix_given_x) + 0.5 * (
            tensor.sum(tensor.log(invsigma_given_x), axis=1) + log_exponent)
max_exponent = tensor.max(lpr, axis=1)
mod_exponent = lpr - max_exponent[:, None]
gauss_mix = tensor.sum(tensor.exp(mod_exponent), axis=1)
log_gauss = tensor.log(gauss_mix) + max_exponent
res = - tensor.mean(log_gauss)

mdn_gradient = []
for param in mdn_model.params:
    mdn_gradient.append(tensor.grad(cost=mdn_nll, wrt=param))


model_gradients = theano.function(
        inputs=[index],
        outputs=mdn_gradient,
    givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


mdn_updates = []
for param in mdn_model.params:
    param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
    mdn_updates.append((param, param - learning_rate * param_update))
    mdn_updates.append((param_update, momentum*param_update + (1. - momentum)*tensor.grad(cost=mdn_nll, wrt=param)))


gparams = [tensor.grad(mdn_nll, param) for param in mdn_model.params]
mdn_updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(mdn_model.params, gparams)
    ]

Wparams = [mdn_model.params[0], mdn_model.params[2]]
gparams = [tensor.grad(mdn_nll, param) for param in Wparams]
mdn_updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(Wparams, gparams)
    ]

# mdn_updates = []
# for param in mdn_model.params:
#     mdn_updates.append((param, param - learning_rate * tensor.grad(cost=mdn_nll, wrt=param)))

model_gradients = theano.function(
    inputs=[index],
    outputs=gparams,
    givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

train_model = theano.function(
        inputs=[index],
        outputs=mdn_nll,
        updates=mdn_updates,
    givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


train_model(0)

print('... training the model')
patience = 5000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
# found
improvement_threshold = 0.995  # a relative improvement of this much is
# considered significant
validation_frequency = min(n_train_batches, patience / 2)

best_validation_loss = numpy.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0
n_epochs = 50

test_model = theano.function(
    inputs=[index],
    outputs=mdn_nll,
    givens={
        x: test_set_x[index * batch_size:(index + 1) * batch_size],
        y: test_set_y[index * batch_size:(index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=mdn_nll,
    givens={
        x: valid_set_x[index * batch_size:(index + 1) * batch_size],
        y: valid_set_y[index * batch_size:(index + 1) * batch_size]
    }
)


while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_batches)]
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
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)
                best_validation_loss = this_validation_loss
                # test it on the test set
                test_losses = [test_model(i)
                               for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)
                print(
                    (
                        '     epoch %i, minibatch %i/%i, test error of'
                        ' best model %f %%'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.
                    )
                )
                # save the best model
                with open('best_model.pkl', 'w') as f:
                    cPickle.dump(mdn_model, f)
        if patience <= iter:
            done_looping = True
            break
end_time = timeit.default_timer()
print(
    (
        'Optimization complete with best validation score of %f %%,'
        'with test performance %f %%'
    )
    % (best_validation_loss * 100., test_score * 100.)
)
