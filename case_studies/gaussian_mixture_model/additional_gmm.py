"""
Code from the basic GMM implementation, reproduced here to reduce visual clutter
"""

import jax.numpy as jnp
import jax
import numpy as np  
np.random.seed(123)

from sklearn.datasets import make_blobs
# We use a jit to speed up the function. a `jit` + `vmap` should achieve the same performance
@jax.jit
def gaussian_pdf(coor: jnp.array, mu_k: jnp.array, sigma_k: jnp.array) -> jnp.array:
    # PDF formula from: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    k = len(mu_k)
    t1 = (2 * jnp.pi) ** (-k / 2)
    t2 = jnp.linalg.det(sigma_k) ** (-0.5)

    inv = jnp.linalg.inv(sigma_k)
    diff = coor - mu_k
    to_exp = -0.5 * jnp.sum(diff @ inv * diff, axis=1)

    to_ret = t1 * t2 * jnp.exp(to_exp)

    assert len(to_ret) == len(coor)
    return to_ret

def log_likelihood(data, mu, sigma, pi, K):
    log_likelihood = 0
    for data_point in data:
        mixture_likelihood = 0
        for k in range(K):
            v = pi[k] * gaussian_pdf(
                jnp.expand_dims(data_point, axis=0), mu_k=mu[k], sigma_k=sigma[k]
            )
            mixture_likelihood += v
        log_likelihood += jnp.log(mixture_likelihood)

    return log_likelihood


@jax.jit
def _loglikelihood_gaussian(x_i: jnp.array, cls_prior_k: jnp.array, mu_k: jnp.array, sigma_k: jnp.array) -> jnp.array:
    """
    Calculate the LL for a single point
    Args:
        x_i: vector of shape (1, num_feats)
        mu_k: vector of shape (1, num_feats)
        sigma_k: matrix of shape (num_feats, num_feats)

    """
    k = len(mu_k)
    sigma_inv = jnp.linalg.inv(sigma_k)
    sigma_det = jnp.linalg.det(sigma_k)
    log_det_sigma = jnp.log(sigma_det)

    diff = x_i - mu_k
    t1 = -0.5 * k * jnp.log(2 * jnp.pi)
    t2 = -0.5 * log_det_sigma
    t3 = -0.5 * diff @ sigma_inv @ diff

    return t1 + t2 + t3 + jnp.log(cls_prior_k)


@jax.jit
def ll_gaussian(X, cls_prior_k, mu_k, sigma_k):
    return jax.vmap(
        fun=_loglikelihood_gaussian,
        in_axes=(0, None, None, None)
    )(X, cls_prior_k, mu_k, sigma_k)

@jax.jit
def calculate_normalizer(log_prob_arr: jnp.ndarray) -> jnp.ndarray:
    """
    For the log-sum-exp
    Args:
        log_prob_arr:

    Returns:

    """
    _max = jnp.max(log_prob_arr, axis=0)
    return _max + jnp.log(jnp.sum(
        jnp.exp(log_prob_arr - _max),
        axis=0
    ))


@jax.jit
def e_step(X, mus, sigmas, cls_priors):
    ll_gaussian_over_parameters = jax.vmap(
        ll_gaussian,
        in_axes=(None, 0, 0, 0)
    )
    _responsibilities = ll_gaussian_over_parameters(X, cls_priors, mus, sigmas)
    normalizer = calculate_normalizer(_responsibilities)
    return jnp.exp(_responsibilities - normalizer).T[0]

@jax.jit
def _m_step_single(x_i, mu_k, resp_nk):
    """
    Calculate the individual values for mu and sigma
    """

    mu_new = x_i * resp_nk
    diff = x_i - mu_k
    sigma_new = resp_nk * jnp.outer(diff, diff)
    return mu_new, sigma_new

@jax.jit
def _m_step(X, mu_k, resp_k):
    N_k = jnp.sum(resp_k)
    to_ave_mus, to_ave_sigmas = jax.vmap(
        _m_step_single,
        in_axes=(0, None, 0)
    )(X, mu_k, resp_k)

    mus = jnp.sum(to_ave_mus, axis=0) / N_k
    sigmas = jnp.sum(to_ave_sigmas, axis=0) / N_k
    cls_prior = N_k / len(X)
    return mus, sigmas, cls_prior

@jax.jit
def m_step(X, mus, responsibilities):
    mus, sigmas, cls_prior = jax.vmap(
        _m_step,
        in_axes=(None, 0, 1)
    )(X, mus, responsibilities)

    return mus, sigmas, jnp.expand_dims(cls_prior, -1)

def gmm_single_iteration():
    pass

def EM_GMM(
        data: np.ndarray,
        guess_num_classes,
        # Initial guesses
        mus, sigmas, cls_probs,

        verbose=False

):
    counter = 0
    ll_container = []
    TOL = 0.00001
    ll_container.append(jnp.inf)
    jax.lax.while_loop(

    )

    while True:  # Run until converges
        # e-step
        responsibilities = e_step(data, mus, sigmas, cls_probs)

        # m-step
        mus, sigmas, cls_probs = m_step(data, mus, responsibilities)
        # Recalculate the log-likelihood
        ll_curr = log_likelihood(data, mus, sigmas, cls_probs, guess_num_classes)

        if jnp.abs(ll_container[-1] - ll_curr) < TOL:
            jax.debug.print("Converged to within {TOL} after: {counter} iterations", TOL=TOL, counter=counter)
            break

        ll_container.append(ll_curr)
        if verbose:
            jax.debug.print("Converged to within {TOL} after: {counter} iterations", TOL=TOL, counter=counter)
            jax.debug.print("Data Log-Likelihood at iteration: {counter} = {ll_curr}", counter=counter, ll_curr=ll_curr)
        counter += 1

    responsibilities = e_step(data, mus, sigmas, cls_probs)
    return mus, sigmas, cls_probs.T, responsibilities.T, ll_container[1:]

unknown_centers = np.asarray([
    [1, -1],  # bottom left
    [5, 5],  # middle
    [8, 7],  # mid-right
    [10, 0]  # bottom right
])

def make_ds(centers):
    points_in_classes = [30, 50, 20, 5]
    ################################################
    # Initial Guesses
    ################################################
    # Randomly increase/ decrease by 10% each way
    scale = (np.random.randint(low=0, high=20, size=centers.shape) - 10) / 100

    initial_mu_guesses = centers + (centers * scale)
    return make_blobs(points_in_classes, centers=centers), initial_mu_guesses
