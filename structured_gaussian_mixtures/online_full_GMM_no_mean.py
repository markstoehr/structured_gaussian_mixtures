from __future__ import division, print_function
from bm_tools import OnlineLogsumexp, sigmoid, log1pexp, logsumexp
import numpy
from scipy import linalg

TRAIN = "/home/mark/Projects/succotash/succotash/datasets/train_examples.npy"
TEST = "/home/mark/Projects/succotash/succotash/datasets/test_examples.npy"

X = numpy.load(TRAIN)

n_components = 24
n_features = X.shape[1]
model_weights = numpy.ones(n_components)/n_components
model_covars = numpy.zeros((n_components,n_features, n_features))

second_moment_stats = numpy.zeros((n_components, n_features, n_features))
weights = numpy.zeros(model_weights.shape)

def score_samples(X,model_weights,model_covars,use_scipy_misc=False):
    n_samples, n_dim = X.shape
    nmix = len(model_covars)
    lpr = numpy.empty((n_samples, nmix))
    for c, cv in enumerate(model_covars):
        cv_chol = linalg.cholesky(cv, lower=True)
        cv_log_det = 2 * numpy.sum(numpy.log(numpy.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, X.T, lower=True).T
        lpr[:, c] = - .5 * (numpy.sum(cv_sol ** 2, axis=1) +
                                 n_dim * numpy.log(2 * numpy.pi) + cv_log_det)
    if numpy.any(numpy.isnan(lpr)):
        import pdb; pdb.set_trace()
    lpr += numpy.log(model_weights)
    logprob = logsumexp(lpr)
    responsibilities = numpy.exp(lpr - logprob[:,numpy.newaxis])
    return logprob, responsibilities

for i in xrange(n_components):
    # model_means[i] = numpy.mean(X[i::n_components],0)
    model_covars[i] = numpy.cov(X[i::n_components].T)

second_moment_stats[:] = cv
minibatch_size = 300
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
        X_batch, model_weights,model_covars)
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
    inverse_weights = 1.0/(weights_tmp[:,numpy.newaxis] + 1e-7)
    model_weights = weights/(weights.sum() + 1e-5) + 1e-6
    model_weights /= model_weights.sum()
    # model_means[:] = first_moment_stats    
    cv = numpy.empty((n_components, n_features, n_features))
    for c in range(n_components):
        post = responsibilities[:, c]
        # Underflow Errors in doing post * X.T are  not important
        numpy.seterr(under='ignore')
        cv[c] = numpy.dot(post * X_batch.T, X_batch) * (inverse_weights[c])
        print(c, numpy.linalg.slogdet(cv[c]))
    second_moment_stats[:] += alpha * (cv - second_moment_stats)
    model_covars = second_moment_stats
    print(current_log_likelihood,i)


X_test = numpy.load(TEST)
lls, responsibilities = score_samples(X_test,model_weights,model_covars)
print(n_components, lls.mean())
# n_c = 25; lls = 
# n_c = 50; lls = 
# n_c = 100; lls = -1764.16076962
# n_c = 200; lls = -1761.26919623
# n_c = 300; lls = -1761.63032148

from sklearn.mixture import gmm

smodel = gmm.GMM(n_components=24, covariance_type='full', n_iter=200)
smodel.fit(X)
smodel.score(X_test).mean()

# n_c = 24; lls = -1686.6809502524477
