from __future__ import division, print_function
from bm_tools import OnlineLogsumexp, sigmoid, log1pexp, logsumexp
import numpy

TRAIN = "/home/mark/Projects/succotash/succotash/datasets/train_examples.npy"
TEST = "/home/mark/Projects/succotash/succotash/datasets/test_examples.npy"

X = numpy.fft.fft(numpy.load(TRAIN))
X /= numpy.sqrt(X.shape[1])

n_components = 200
n_features = X.shape[1]
model_means = numpy.zeros((n_components,n_features))
model_weights = numpy.ones(n_components)/n_components
model_covars = numpy.zeros((n_components,n_features))

first_moment_stats = numpy.zeros(model_means.shape)
second_moment_stats = numpy.zeros(model_means.shape)
weights = numpy.zeros(model_weights.shape)

def score_samples(X,model_weights,model_covars,use_scipy_misc=False):
    inv_covars = 1.0/model_covars
    n_features = X.shape[1]
    lpr= - 0.5 * ( n_features * numpy.log(2*numpy.pi) +
                    numpy.sum(numpy.log(model_covars),1)
                    + numpy.dot(X**2, inv_covars.T))
    if numpy.any(numpy.isnan(lpr)):
        import pdb; pdb.set_trace()
    lpr += numpy.log(model_weights)
    logprob = logsumexp(lpr)
    responsibilities = numpy.exp(lpr - logprob[:,numpy.newaxis])
    return logprob, responsibilities

for i in xrange(n_components):
    # model_means[i] = numpy.mean(X[i::n_components],0)
    model_covars[i] = numpy.mean((X[i::n_components] - model_means[i])**2,0)


minibatch_size = 400
alpha = 0.05
n_batches = X.shape[0]/minibatch_size

current_log_likelihood=None
for i in xrange(2000):
    prev_log_likelihood = current_log_likelihood
    batch_idx = i % n_batches
    if batch_idx == n_batches - 1:
        batch_end = X.shape[0]
    else:
        batch_end = (batch_idx+1)*minibatch_size
    cur_minibatch_size = batch_end - batch_idx*minibatch_size
    X_batch = X[batch_idx*minibatch_size:batch_end]
    lls, responsibilities = score_samples(
        X_batch,
        model_means,model_weights,model_covars)
    current_log_likelihood = lls.mean()
    if prev_log_likelihood is not None:
        change = abs((current_log_likelihood - prev_log_likelihood)/prev_log_likelihood)
        if change < .00001:
            pass #break
    weights_tmp = responsibilities.sum(0)
    if i == 0:
        weights[:] = weights_tmp
    else:
        weights += alpha * ( weights_tmp - weights)
    weighted_X_sum = numpy.dot(responsibilities.T,X_batch)
    inverse_weights = 1.0/(weights_tmp[:,numpy.newaxis] + 1e-5)
    model_weights = weights/(weights.sum() + 1e-5) + 1e-6
    model_weights /= model_weights.sum()
    if i == 0:
        first_moment_stats[:] = weighted_X_sum * inverse_weights
        second_moment_stats
    else:
        first_moment_stats += alpha * (weighted_X_sum * inverse_weights - first_moment_stats)
    weighted_X_sq_sum = numpy.dot(responsibilities.T,X_batch**2) * inverse_weights
    if i == 0:
        second_moment_stats[:] = weighted_X_sq_sum
    else:
        second_moment_stats[:] += alpha * (weighted_X_sq_sum - second_moment_stats)
    model_covars = second_moment_stats - first_moment_stats**2
    print(current_log_likelihood,i)


X_test = numpy.fft.fft(numpy.load(TEST))
X_test /= numpy.sqrt(X_test.shape[1])
lls, responsibilities = score_samples(X_test,model_means,model_weights,model_covars)
print(lls.mean())
# n_c = 25; lls = -2054.1306788521774
# n_c = 50; lls = -2040.20835419
# n_c = 100; lls = -2039.88325656
# n_c = 200; lls = -2039.63340873

