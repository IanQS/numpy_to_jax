import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np

def plot_data(X, y, num_centers):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    
    ax[0].scatter(X[:, 0], X[:, 1])
    for cluster_idx in range(num_centers):
        mask = y == cluster_idx
        cluster_xs, cluster_ys = X[mask], y[mask]
    
        ax[1].scatter(cluster_xs[:, 0], cluster_xs[:, 1])
    
    
    ax[0].set_title("Unlabeled Data")
    ax[0].set_xlabel("Dim 1")
    ax[0].set_ylabel("Dim 2")
    
    ax[1].set_title("Labeled Data")
    ax[1].set_xlabel("Dim 1")
    ax[1].set_ylabel("Dim 2")
    
    fig.suptitle('Data in Unlabeled and Labeled form')
    
    plt.tight_layout()
    plt.show()


def plot_training(lls):
    plt.plot(lls)
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood of points")
    plt.show()



def confidence_ellipse(mu, sigma, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Modified based on function from: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = sigma[0, 1]/np.sqrt(sigma[0, 0] * sigma[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(sigma[0, 0]) * n_std
    # calculating the standard deviation of y ...
    scale_y = np.sqrt(sigma[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mu[0], mu[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)