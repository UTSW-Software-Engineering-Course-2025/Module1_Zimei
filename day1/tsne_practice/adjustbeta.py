import numpy as np
import torch

def Hbeta(D, beta=1.0):
    r"""
    Compute entropy(H) and probability(P) from nxn distance matrix.

    Parameters
    ----------
    D : numpy.ndarray
        distance matrix (n,n)
    beta : float
        precision measure
    .. math:: \beta = \frac{1}/{(2 * \sigma^2)}

    Returns
    -------
    H : float
        entropy
    P : numpy.ndarray
        probability matrix (n,n)
    
    Examples
    --------
    >>> D = np.array([[0, 1], [1, 0], [2, 2]])
    >>> beta = 1.0
    >>> H, P = Hbeta(D, beta)
    >>> print(H)
    [1.2571852 1.2571852]
    >>> print(P)
    [[0.66524096 0.24472847]
    [0.24472847 0.66524096]
    [0.09003057 0.09003057]]
    """
    num = np.exp(-D * beta)
    den = np.sum(np.exp(-D * beta), 0)
    P = num / den
    H = np.log(den) + beta * np.sum(D * num) / (den)
    return H, P


def adjustbeta(X, tol, perplexity):
    """
    Precision(beta) adjustment based on perplexity

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    tol : float
        tolerance for the stopping criteria of beta adjustment
    perplexity : float
        perplexity can be interpreted as a smooth measure of the effective number of neighbors
    
    Raises
    ----------
        ValueError: If the input tensor is not 2D.

    Returns
    -------
    P : numpy.ndarray
        probability matrix (n,n)
    beta : numpy.ndarray
        precision array (n,1)
    
    Examples
    --------
    >>> X = np.array([[0, 1], [1, 0], [2, 2]])
    >>> tol = 1e-5
    >>> perplexity = 2.0
    >>> P, beta = adjustbeta(X, tol, perplexity)
    >>> print(P)
    [[0.         0.50146484 0.49853516]
    [0.50146484 0.         0.49853516]
    [0.5        0.5        0.        ]]
    >>> print(beta)
    [[0.00195312]
    [0.00195312]
    [1.        ]]
    """
    # Check if X is a 2D arrays
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    (n, d) = X.shape
    # Need to compute D here, which is nxn distance matrix of X
    from tsne import compute_pairwise_dist
    D = compute_pairwise_dist(torch.from_numpy(X).float()).numpy()
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisP

    return P, beta

if __name__ == "__main__":
    # Example usage
    # Hbeta
    D = np.array([[0, 1], [1, 0], [2, 2]])
    beta = 1.0
    H, P = Hbeta(D, beta)
    print("H:", H)
    print("P:", P)
    # adjustbeta
    X = np.array([[0, 1], [1, 0], [2, 2]])
    tol = 1e-5
    perplexity = 2.0
    P, beta = adjustbeta(X, tol, perplexity)
    print("P:", P)
    print("beta:", beta)