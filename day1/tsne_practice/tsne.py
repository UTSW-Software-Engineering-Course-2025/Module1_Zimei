import matplotlib.pyplot as plt
import torch
import numpy as np
from adjustbeta import adjustbeta
def compute_pairwise_dist(X):
    """
    Calculates pairwise squared Euclidean distances for an input tensor.
    
    we caculate the dist_sq[i, j] = ||x_i - x_j||^2
    Parameters
    ----------
    X_tensor : torch.Tensor
        Input data tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor
        Pairwise squared Euclidean distances, shape (n_samples, n_samples).
    
    Raises
    ----------
        ValueError: If the input tensor is not 2D.
    
    Examples
    ----------
    >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> compute_pairwise_dist(X)
    tensor([[ 0.,  8., 32.],
            [ 8.,  0.,  8.],
            [32.,  8.,  0.]])

    """
    if X.ndim != 2:
        raise ValueError("Input tensor must be 2D (n_samples, n_features).")
    
    # Calculate squared norms of each point ||x_i||^2 
    sum_X_sq = torch.sum(X * X, dim=1, keepdim=True)
    # Calculate the distance dist_sq[i, j] = ||x_i - x_j||^2 = ||x_i||^2 - 2 * <x_i, x_j> + ||x_j||^2 
    dist_sq = sum_X_sq - 2 * torch.matmul(X, X.T) + sum_X_sq.T
    # Clamp squared distances to be non-negative
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return dist_sq


def compute_pairwise_affinity(dist_sq, betas):
    """
    Computes conditional (P_j|i) and joint (P_ij) pairwise affinities.
    P_ij is normalized to sum to 1 over all pairs.

    Args:
        dist_sq (torch.Tensor): Input squared distance matrix of shape (n_samples, n_samples).
        betas (torch.Tensor): A 1D tensor of shape (n_samples,) containing
                                  the beta_i =1/(2* sigma_i^2) for each data point x_i.
                                  betas[i] corresponds to the inverse of the variance for point i.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - P_conditional_ji (torch.Tensor): Tensor of shape (n_samples, n_samples) where
                                               P_conditional_ji[i, j] corresponds to p_{j|i}.
                                               Rows sum to 1 (approximately).
            - P_joint (torch.Tensor): Symmetrized joint probabilities, tensor of shape
                                      (n_samples, n_samples) where P_joint[i, j] is p_ij.
                                      The entire matrix sums to 1.
    """
    # n_samples = X.shape[0]


    # (1) Compute P_conditional_ji ($p_{j|i}$)
    # Numerator: exp(-||x_i - x_j||^2 / 2 * sigma_i^2)
    # betas.unsqueeze(1) makes it a column vector for row-wise sigma_i
    numerator_p_ji = torch.exp(-dist_sq * betas)
    # Set diagonal to 0 because we sum over k != i (and j != i for p_j|i)
    numerator_p_ji.fill_diagonal_(0)

    # Denominator: sum_{k != i} exp(-||x_i - x_k||^2 / 2 * sigma_i^2)
    denominator_p_ji = torch.sum(numerator_p_ji, dim=1, keepdim=True)
    print("denominator_p_ji",denominator_p_ji.shape)

    # P_conditional_ji[i, j] = p_{j|i}
    P_conditional_ji = numerator_p_ji / denominator_p_ji
    # Handle cases where denominator might be zero (e.g., sigma_i is extremely small or point is isolated)
    P_conditional_ji = torch.nan_to_num(P_conditional_ji, nan=0.0, posinf=0.0, neginf=0.0)

    P_joint_numerator = P_conditional_ji + P_conditional_ji.T

    # The denominator is the sum of all elements in P_joint_numerator
    # (since diagonals are already zero for P_conditional_ji and P_conditional_ij)
    P_joint_denominator = torch.sum(P_joint_numerator)

    # In t-SNE, this denominator typically simplifies to 2*n_samples.
    # If P_joint_denominator is zero (e.g. all affinities are zero), P_joint should be zero.
    if P_joint_denominator > 0:
        P_joint = P_joint_numerator / P_joint_denominator
    else:
        P_joint = torch.zeros_like(P_joint_numerator)

    # Ensure diagonal is zero for joint probabilities
    P_joint.fill_diagonal_(0)

    return P_conditional_ji, P_joint

def normalize_exaggerate_and_clip(P: torch.Tensor,
                                  min_clip: float = 1e-12,early_ex: float = 4.0
                                    ) -> torch.Tensor:
    """
    Normalizes the joint probabilities P_ij to ensure they sum to 1 (if not already),
    applies early exaggeration, and clips values to a minimum.

    Args:
        P(torch.Tensor): Probabilities tensor of shape (n_samples, n_samples).
        min_clip (float): The minimum value to clip probabilities to.

    Returns:
        torch.Tensor: Exaggerated and clipped joint probabilities tensor.
    """
    Psum = torch.sum(P)
    # Check if P_joint is all zeros, if so, the normalization would be NaN.
    if Psum < 1e-9 : # Handle case where P_joint is all (or nearly all) zeros
        P_norm = torch.zeros_like(P)
    else: 
        P_norm = P / Psum
    # Normalize the joint probabilities

    #early_exaggeration = 4.0
    P_ex = P_norm * early_ex
    # Clip the probabilities
    P_clipped = torch.clamp(P_ex, min=min_clip)

    return P_clipped

def compute_low_dim_affinity(dist_Y):
    """
    Computes the low-dimensional affinity matrix for t-SNE.

    Args:
        dist_Y (torch.Tensor): Pairwise squared Euclidean distances in low-dimensional space.

    Returns:
        torch.Tensor: Low-dimensional affinities.
    """
    # Compute the low-dimensional affinities

    Q_numerator = 1.0/(1.0+dist_Y)
    Q_denominator = torch.sum(Q_numerator)
    # Q_denominator.fill_diagonal_(0)  # Set diagonal to zero 
    # Normalize the low-dimensional affinities
    Q_normalized = Q_numerator / Q_denominator
    # Handle cases where Q might be NaN or Inf
    Q = torch.nan_to_num(Q_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    # # Ensure diagonal is zero for low-dimensional affinities
    Q.fill_diagonal_(0)
    # print("Q",Q)
    # Qsum = torch.sum(Q)
    # if Qsum < 1e-9 :
    #     Q_normalized = torch.zeros_like(Q)
    # else:
    #     Q_normalized = Q / Qsum
    # print("Q_normalized",Q_normalized)
    Q = torch.clamp(Q, min=1e-12)
    return Q
def compute_kl_divergence(P, Q):
    """
    Computes the Kullback-Leibler divergence between the high-dimensional and low-dimensional distributions.

    Args:
        P(torch.Tensor): Joint probabilities tensor of shape (n_samples, n_samples).
        Q (torch.Tensor): Low-dimensional affinities tensor of shape (n_samples, n_samples).

    Returns:
        torch.Tensor: Kullback-Leibler divergence.
    """
    # Compute KL divergence
    kl_div = torch.sum(P * torch.log(P / Q), dim=1)
    return kl_div

def compute_gradient_loss_fucntion(P,Q,Y):
    """
    Computes the gradient of the loss function for t-SNE.

    Args:
        P(torch.Tensor): Joint probabilities tensor of shape (n_samples, n_samples).
        Q (torch.Tensor): Low-dimensional affinities tensor of shape (n_samples, n_samples).
        Y (torch.Tensor): Low-dimensional representation of the data.
    Returns:
        torch.Tensor: Gradient of the loss function.
    """
    # Compute pairwise squared distances in low-dimensional space
    dist_y = compute_pairwise_dist(Y) # (n_samples, n_samples)
    print(dist_y.shape)
    # compute the kernel
    k = 1.0/(1.0+dist_y) 
    print(k.shape)
    # compute the pairwise difference
    pdiff = P - Q
    print(pdiff.shape)
    print(Y.shape)
    # compute the difference of y
    diff_y = Y.unsqueeze(1) - Y.unsqueeze(0)
    print(diff_y.shape)
    # Compute the gradient of the loss function
    dY = torch.einsum('ij,ijk,ij->ik', pdiff, diff_y, k) 

    return dY

def tsne(X, low_dims =2, perplexity=30.0, initial_p=0.5, final_p=0.8, eta=500, min_gain=0.01, T =1000):
    """
    Computes t-SNE embedding for the input data.
    Args:
        X (numpy.narray): Input data tensor of shape (n_samples, n_features).
        low_dims (int): Number of dimensions for the output embedding.
        perplexity (float): Perplexity parameter for t-SNE.
        initial_p (float): Initial momentum.
        final_p (float): Final momentum.
        eta (float): Learning rate.
        min_gain (float): Minimum gain for gradient updates.
        T (int): Number of iterations.
    """
    tol = 1e-5
    early_ex = 4
    _, betas = adjustbeta(X, tol, perplexity)
    print(betas)
    betas = torch.from_numpy(betas).float().to("cuda:0")
    X = torch.from_numpy(X).float().to("cuda:0")
    # Compute pairwise squared distances
    dist_X = compute_pairwise_dist(X)
    # Compute pairwise affinities
    P_ji, P_joint = compute_pairwise_affinity(dist_X, betas)
    # Normalize and exaggerate P_joint
    P_joint = normalize_exaggerate_and_clip(P_joint,early_ex=early_ex)

    # Initialize low-dimensional representation
    Y = torch.from_numpy(pca(X.cpu().numpy(), low_dims = low_dims)).to("cuda:0")
    delta_Y = torch.zeros_like(Y)
    gain = torch.ones_like(Y)
    print("P_joint",P_joint)
    print("Y",Y)
    for t in range(1):
        # Compute pairwise squared distances in low-dimensional space
        dist_Y = compute_pairwise_dist(Y)
        # Compute low-dimensional affinities
        Q = compute_low_dim_affinity(dist_Y)
        # Normalize and clip Q
        Q = torch.clamp(Q, min=1e-12)
        # Compute gradient of the loss function
        dY = compute_gradient_loss_fucntion(P_joint, Q, Y)
        if t < 20:
            momentum = initial_p
        else:
            momentum = final_p
        
        # Update gain and delta_Y
        # Update gain based on the sign of dY
        # If dY > 0, increase gain; if dY < 0, decrease gain
        gain = (gain + 0.2) * ((dY > 0) != (delta_Y>0)) + (gain * 0.8) * ((dY > 0) == (delta_Y>0)) 
        gain = torch.clamp(gain, min=min_gain)
        # Update delta_Y based on the momentum and gain
        delta_Y = momentum * delta_Y - eta * gain * dY
        Y += delta_Y
        if t == 100:
            P_joint = P_joint / early_ex
    print("Q",Q)
    print("dY",dY)
    

    return Y

def pca(X, low_dims=50):
    """
    Runs PCA on the nxd array X in order to reduce its dimensionality to
    low_dims dimensions.

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    low_dims : int
        number of dimensions that PCA reduce to

    Returns
    -------
    Y : numpy.ndarray
        low-dimensional representation of input X
    """
    n, d = X.shape
    X = X - X.mean(axis=0)[None, :]
    _, M = np.linalg.eig(np.dot(X.T, X))
    Y = np.real(np.dot(X, M[:, :low_dims]))
    return Y


if __name__ == "__main__":
    print("Run Y = tsne(X, low_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt('mnist2500_X.txt')
    print("X shape:", X.shape)
    X = pca(X, 50)
    labels = np.loadtxt('mnist2500_labels.txt')
    print("labels shape:", labels.shape)
    Y = tsne(X , perplexity=30.0, eta=500)
    Y = Y.cpu().numpy()
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.savefig("mnist_tsne.png")
