"""Estimate a diagonal GMM and estimate the log-likelihood."""
from __future__ import division, print_function
from bm_tools import OnlineLogsumexp, sigmoid, log1pexp, logsumexp
import numpy

TRAIN = "/home/mark/Projects/succotash/succotash/datasets/train_examples.npy"
TEST = "/home/mark/Projects/succotash/succotash/datasets/test_examples.npy"

X = numpy.load(TRAIN)

n_components = 25
n_features = X.shape[1]
model_means = numpy.zeros((n_components,n_features))
model_weights = numpy.ones(n_components)/n_components
model_covars = numpy.zeros((n_components,n_features))

def score_samples(X,model_means,model_weights,model_covars,use_scipy_misc=False):
    inv_covars = 1.0/model_covars
    n_features = X.shape[1]
    lpr= - 0.5 * ( n_features * numpy.log(2*numpy.pi) +
                    numpy.sum(numpy.log(model_covars),1)
                    + numpy.sum( (model_means**2)*inv_covars,1)
                    - 2 * numpy.dot(X, (model_means*inv_covars).T)
                    + numpy.dot(X**2, inv_covars.T))
    if numpy.any(numpy.isnan(lpr)):
        import pdb; pdb.set_trace()
    lpr += numpy.log(model_weights)
    logprob = logsumexp(lpr)
    responsibilities = numpy.exp(lpr - logprob[:,numpy.newaxis])
    return logprob, responsibilities

for i in xrange(n_components):
    model_means[i] = numpy.mean(X[i::n_components],0)
    model_covars[i] = numpy.mean((X[i::n_components] - model_means[i])**2,0)

current_log_likelihood=None
for i in xrange(600):
    prev_log_likelihood = current_log_likelihood
    lls, responsibilities = score_samples(X,model_means,model_weights,model_covars)
    current_log_likelihood = lls.mean()
    if prev_log_likelihood is not None:
        change = abs((current_log_likelihood - prev_log_likelihood)/prev_log_likelihood)
        if change < .00001:
            pass #break
    weights = responsibilities.sum(0)
    weighted_X_sum = numpy.dot(responsibilities.T,X)
    inverse_weights = 1.0/(weights[:,numpy.newaxis] + 1e-5)
    model_weights = weights/(weights.sum() + 1e-5) + 1e-6
    model_weights /= model_weights.sum()
    model_means = weighted_X_sum * inverse_weights
    model_covars = numpy.dot(responsibilities.T, X * X) * inverse_weights - 2 * model_means * weighted_X_sum * inverse_weights + model_means**2 
    print(current_log_likelihood,i)



X_test = numpy.load(TEST)
lls, responsibilities = score_samples(X_test,model_means,model_weights,model_covars)
# test n_c = 12; ll = -2073.4514191
# test n_c = 25; ll = -2057.64

