from __future__ import division, print_function
from theano import tensor
import theano
import mdn
import numpy


TRAIN = "/home/mark/Projects/succotash/succotash/datasets/train_double_examples.npy"
TEST = "/home/mark/Projects/succotash/succotash/datasets/train_double_test_examples.npy"

# X = numpy.fft.fft(numpy.load(TRAIN))
# X /= numpy.sqrt(X.shape[1])
Z = numpy.load(TRAIN)
Z_test = numpy.load(TEST)

WINSIZE = Z.shape[1]//2
X = theano.shared(Z[:,:WINSIZE].astype(dtype=numpy.float32), borrow=True)
Y = theano.shared(Z[:,WINSIZE:].astype(dtype=numpy.float32), borrow=True)

X_test = theano.shared(Z_test[:,:WINSIZE].astype(dtype=numpy.float32), borrow=True)
Y_test = theano.shared(Z_test[:,WINSIZE:].astype(dtype=numpy.float32), borrow=True)

x = tensor.fmatrix('x')
y = tensor.fmatrix('y')

b_values = np.zeros((WINSIZE,n_components), dtype=theano.config.floatX)
        
W_values = np.asarray(rng.uniform(
    low=-np.sqrt(6. / (n_in + n_out)),
    high=np.sqrt(6. / (n_in + n_out)),
    size=(WINSIZE, WINSIZE, n_components)),
                      dtype=theano.config.floatX)

W_sigma = theano.shared(value=W_values, name='W_sigma',
                                     borrow=True)
W_mixing = theano.shared(value=W_values[:,:,0].copy(), name='W_mixing',
                                      borrow=True)

b_sigma = theano.shared(value=b_values.copy(), name='b_sigma',
                             borrow=True)
b_mixing = theano.shared(value=b_values[0].copy(), name='b_mixing',
                              borrow=True)
sigma = T.nnet.softplus(T.dot(input, W_sigma)) #+\
             #b_sigma.dimshuffle('x',0))
mixing = T.nnet.softmax(T.dot(input, W_mixing)) #+\
              #b_mixing.dimshuffle('x',0))

class NetworkLayer



hid_activations = [tensor.tanh, tensor.tanh]
n_hiddens = [300,300]

rng = numpy.random.RandomState(1234)
frame_pred = mdn.MDN(rng=rng, input=x,
                     n_in=WINSIZE,
                     n_hiddens=n_hiddens,
                     hid_activations=hid_activations,
                     n_out=WINSIZE,
                     out_activation=None,
                     n_components=5)

frame_cost = mdn.NLL(sigma=frame_pred.outputLayer.sigma,
                     mixing=frame_pred.outputLayer.mixing,
                     y=y)


learning_rate = 0.001
n_epochs = 50
batch_size = 300
(total_training_costs,
 total_validation_costs,
 total_validation_MSE) = (
     frame_pred.train(y=y, training_loss=frame_cost,
                      learning_rate=learning_rate,
                      n_epochs=n_epochs,
                      train_x=X, train_y=Y,
                      valid_x=X_test,
                      valid_y=Y_test,
                      batch_size=batch_size))
