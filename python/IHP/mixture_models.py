import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.stats import multivariate_normal
from progress.bar import ChargingBar

class SoftmaxMixture():
    def __init__(self, num_clusters, num_features, global_op = True, eps = 1e-4):
        self._num_clusters = num_clusters
        self._num_features = num_features
        self._global_op = global_op
        self._W = np.random.randn(num_clusters, num_features)
        self._pi = np.ones(num_clusters)/num_clusters
        self._eps = eps
        self.prior = multivariate_normal(mean = np.zeros(self._num_features),
                                        cov=np.eye(self._num_features))
    
    def set_data(self, data):
        self._data = data
        self._N = len(data)
        
    def E(self):
        assert self._data is not None, "Please fix the dataset using the set_data function first"
        Z = np.zeros((self._N, self._num_clusters))
        self._log_probs = np.zeros(self._N)
        bar = ChargingBar('Expectation', max=self._N)
        for n in range(self._N):
            log_likelihoods = self._data[n].compute_log_likelihoods(self._W)
            required_probs = np.exp(log_likelihoods + np.log(self._pi))
            likelihood_sum = np.sum(required_probs)
            self._log_probs[n] = likelihood_sum
            Z[n] = required_probs/likelihood_sum
            bar.next()
        bar.finish()
        print("\n")
        self._Z = Z
        return Z
    
    def _compute_total_error_grad(self, W, Z, k):
        negative_data_grads = np.zeros((self._N, self._num_features))
        negative_log_probs = np.zeros(self._N)
        for n in range(self._N):
            negative_log_probs[n], negative_data_grads[n] = self._data[n].compute_error_grad(W)       
        E = np.dot(Z[:, k], negative_log_probs)
        E_grad = np.zeros(self._num_features)
        for n in range(self._N):
            E_grad += Z[n][k]*negative_data_grads[n]
            
        # Using a prior seems to be important emperically
        E -= self.prior.logpdf(W)
        E_grad += W
        return E, E_grad
    
    def _optimize_params(self, Z):
        W = np.random.randn(self._num_clusters, self._num_features)
        # New Pi
        Z_sum = np.sum(Z, axis = 0)
        self._pi = Z_sum/self._N
        # New Ws using scipy optimize with provided gradient information
        bar = ChargingBar('Maximization', max=self._num_clusters)
        for k in range(self._num_clusters):
            func = lambda x: self._compute_total_error_grad(x, Z, k)
            if self._global_op:
                minimizer_kwargs = {'jac': True}
                res = basinhopping(func, W[k], minimizer_kwargs=minimizer_kwargs)
            else:
                res = minimize(func, W[k], jac = True)
            W[k] = res.x
            bar.next()
        bar.finish()
        self._W = W.reshape(-1, self._num_features)
        
    def M(self, Z):
        self._optimize_params(Z)
    
    def compute_true_ll(self):
        dot_ps = self._log_probs
        dot_ps = [dot_p if dot_p != 0 else self._eps for dot_p in dot_ps]
        ll = np.sum(np.log(dot_ps))
        return ll
    
    def get_params(self):
        return self._W, self._pi, self._Z

class GaussianMixture():
    def __init__(self, K, D):
        self._K = K
        self._D = D
        self._mus = np.random.randn(K, D)
        self._covs = np.array([np.eye(D)] * K)
        self._pi = np.ones(K)/K

    def set_data(self, data):
        self._data = data
        self._N = len(data)
    
    def _compute_likelihood(self, X, mu, cov):
        return multivariate_normal.pdf(X, mu, cov)
    
    def _compute_likelihoods(self, X):
        return [self._compute_likelihood(X, mu, cov) for mu, cov in zip(self._mus, self._covs)]
        
    def E(self):
        assert self._data is not None, "Please fix the dataset using the set_data function first"
        Z = np.zeros((self._N, self._K))
        for n in range(self._N):
            likelihoods = self._compute_likelihoods(self._data[n])
            required_probs = np.multiply(self._pi, likelihoods)
            likelihood_sum = np.sum(required_probs)
            Z[n] = required_probs/likelihood_sum
        return Z
        
    def M(self, Z):
        Z_sum = np.sum(Z, axis = 0)
        self._pi = Z_sum/self._N
        self._mus = np.dot(Z.T, self._data) / Z_sum[:, np.newaxis]
        self._covs = np.empty((self._K, self._D, self._D))
        for k in range(self._K):
            diff = self._data - self._mus[k]
            self._covs[k] = np.dot(Z[:, k] * diff.T, diff) / Z_sum[k]
    
    def compute_true_ll(self):
        ll = 0
        for d in self._data:
            ll += np.log(np.dot(self._compute_likelihoods(d), self._pi))
        return ll
    
    def get_params(self):
        return self._mus, self._covs, self._pi
    
    def reset_params(self):
        self._mus = np.random.randn(self._K, self._D)
        self._covs = np.array([np.eye(self._D)] * self._K)
        self._pi = np.ones(self._K)/self._K

        
class MixtureEM():
    def __init__(self, model, tol = 1e-4, tol_change=1e-5, niters = None):
        self._model = model
        self._tol = tol
        self._tol_change = tol_change
        self._niters = niters
        self._max_iters = 200
    
    def optimize(self, data, global_op = True, predict=True):
        self._model.set_data(data)
        log_diff = 10
        log_relative_change = 10
        iter_count = 0
        while(1):
            print(f"\nIteration: {iter_count}")
            # E-step
            Z = self._model.E()
            # M-step
            self._model.M(Z)
            # Evidence Lower Bound
            ll = self._model.compute_true_ll()
            if iter_count != 0:
                log_diff = ll - previous_ll
                log_relative_change = abs(log_diff/ll)
            previous_ll = ll
            print(f"Log diff is {log_diff}")
            print(f"Log relative change is {log_relative_change}")
            print(f"Current log-likelihood is {ll}")
            iter_count += 1

            if self._niters:
                if iter_count == self._niters:
                    break
            else:
                if log_diff < self._tol or iter_count == self._max_iters or \
                   log_relative_change < self._tol_change:
                    break
        labels = None
        if predict:
            labels = np.argmax(Z, axis=1)

        return self._model, labels, ll
