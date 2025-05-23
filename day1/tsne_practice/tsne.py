

import matplotlib.pyplot as plt
import torch
import numpy as np
from adjustbeta import adjustbeta
import tqdm
import warnings
def compute_pairwise_dist(X):
    """
    Calculates pairwise squared Euclidean distances for an input tensor.
    
    we caculate the dist_sq[i, j] = ||x_i - x_j||^2
    
    Parameters
    ----------
    X : torch.Tensor
        Input data tensor of shape (n_samples, n_features).

    Returns
    ----------
    dist_sq : torch.Tensor
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

    Parameters
    ----------
    dist_sq : torch.Tensor
        Input squared distance matrix of shape (n_samples, n_samples).
    betas : torch.Tensor
        A 1D tensor of shape (n_samples,) containing
        the beta_i =1/(2* sigma_i^2) for each data point x_i.
        betas[i] corresponds to the inverse of the variance for point i.

    Returns
    -----------
    P_joint : torch.Tensor
        Symmetrized joint probabilities, tensor of shape
        (n_samples, n_samples) where P_joint[i, j] is p_ij.
    
    Raises
    ----------
        ValueError: If the input tensor is not 2D.

    Examples
    ----------
    >>> dist_sq = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> betas = torch.tensor([1.0, 1.0])
    >>> compute_pairwise_affinity(dist_sq, betas)
    tensor([[0.5000, 0.5000],
            [0.5000, 0.5000]])
    """
    # (1) Compute P_conditional_ji ($p_{j|i}$)
    # Numerator: exp(-||x_i - x_j||^2 / 2 * sigma_i^2)
    # betas.unsqueeze(1) makes it a column vector for row-wise sigma_i
    numerator_p_ji = torch.exp(-dist_sq * betas)
    # Set diagonal to 0 because we sum over k != i (and j != i for p_j|i)
    numerator_p_ji.fill_diagonal_(0)

    # Denominator: sum_{k != i} exp(-||x_i - x_k||^2 / 2 * sigma_i^2)
    denominator_p_ji = torch.sum(numerator_p_ji, dim=1, keepdim=True)
    
    # P_conditional_ji[i, j] = p_{j|i}
    P_conditional_ji = numerator_p_ji / denominator_p_ji
    # Handle cases where denominator might be zero (e.g., sigma_i is extremely small or point is isolated)
    P_conditional_ji = torch.nan_to_num(P_conditional_ji, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure diagonal is zero
    P_conditional_ji.fill_diagonal_(0)  
    
    # (2) Compute P_joint ($p_{ij}$)
    P_joint_numerator = P_conditional_ji + P_conditional_ji.T

    # The denominator is the sum of all elements in P_joint_numerator
    P_joint_denominator = torch.sum(P_joint_numerator)
    # If P_joint_denominator is zero (e.g. all affinities are zero), P_joint should be zero.
    if P_joint_denominator > 0:
        P_joint = P_joint_numerator / P_joint_denominator
    else:
        P_joint = torch.zeros_like(P_joint_numerator)

    # Ensure diagonal is zero for joint probabilities
    P_joint.fill_diagonal_(0)

    return P_joint

def normalize_exaggerate_and_clip(P: torch.Tensor,
                                  min_clip: float = 1e-12,
                                  early_ex: float = 4.0
                                    ) -> torch.Tensor:
    """
    Normalizes the joint probabilities P_ij to ensure they sum to 1 (if not already),
    applies early exaggeration, and clips values to a minimum.

    Parameters
    ----------
    P : torch.Tensor
        Probabilities tensor of shape (n_samples, n_samples).
    min_clip : float
        The minimum value to clip probabilities to.
    early_ex : float
        Early exaggeration factor to amplify the joint probabilities.

    Returns
    ----------
    P_clipped : torch.Tensor
        Normed, Exaggerated and clipped joint probabilities tensor.

    Raises
    ----------
        ValueError: If the input tensor is not 2D.

    Examples
    ----------
    >>> P = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    >>> normalize_exaggerate_and_clip(P)
    tensor([[0.4000, 0.8000],
        [1.2000, 1.6000]])
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

    Parameters
    ----------
    dist_Y : torch.Tensor
        Pairwise squared Euclidean distances in low-dimensional space.

    Returns:
    ----------
    dist_Y : torch.Tensor
        Low-dimensional affinities.
    
    Examples:
    ----------
    >>> dist_Y = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> compute_low_dim_affinity(dist_Y)
    tensor([[1.0000e-12, 1.6667e-01],
            [1.6667e-01, 1.0000e-12]])
    """
    # Compute the low-dimensional affinities
    Q_numerator = 1.0/(1.0+dist_Y)
    Q_denominator = torch.sum(Q_numerator)

    # Normalize the low-dimensional affinities
    Q_normalized = Q_numerator / Q_denominator
    # Handle cases where Q might be NaN or Inf
    Q = torch.nan_to_num(Q_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    # # Ensure diagonal is zero for low-dimensional affinities
    Q.fill_diagonal_(0)
    # Clip the low-dimensional affinities to avoid numerical issues
    Q = torch.clamp(Q, min=1e-12)
    return Q
def compute_kl_divergence(P, Q):
    """
    Computes the Kullback-Leibler divergence between the high-dimensional and low-dimensional distributions.

    Parameters
    ----------
    P: torch.Tensor
        Joint probabilities tensor of shape (n_samples, n_samples).
    Q: torch.Tensor
        Low-dimensional affinities tensor of shape (n_samples, n_samples).

    Returns:
    -----------
    kl_div : torch.Tensor
        KL divergence.
    """
    # Compute KL divergence
    kl_div = torch.sum(P * torch.log(P / Q), dim=1)
    return kl_div

def compute_gradient_loss_fucntion(P,Q,Y):
    """
    Computes the gradient of the loss function for t-SNE.

    Parameters
    ----------
    P: torch.Tensor
        Joint probabilities tensor of shape (n_samples, n_samples).
    Q : torch.Tensor
        Low-dimensional affinities tensor of shape (n_samples, n_samples).
    Y : torch.Tensor 
        Low-dimensional representation of the data.
    Returns:
    ----------
    dY : torch.Tensor
        Gradient of the loss function.
    """
    # Compute pairwise squared distances in low-dimensional space
    dist_y = compute_pairwise_dist(Y) # (n_samples, n_samples)
    # compute the kernel
    k = 1.0/(1.0+dist_y) 
    # compute the pairwise difference
    pdiff = P - Q
    # compute the difference of y
    diff_y = Y.unsqueeze(1) - Y.unsqueeze(0)
    # Compute the gradient of the loss function
    dY = torch.einsum('ij,ijk,ij->ik', pdiff, diff_y, k) 

    return dY

def get_pytorch_device(device_preference: str | None = None, verbose: bool = True) -> torch.device:
    """
    Determines and returns the appropriate torch.device based on availability
    and user preference.

    Parameters
    ----------
    device_preference : str or None, optional
        User's device preference (e.g., "cuda", "cuda:0", "cuda:1", "cpu", "mps").
        If None, attempts to use the first available CUDA device ('cuda:0'),
        otherwise falls back to CPU. Default is None.
    verbose : bool, optional
        If True, prints information about the device selection and warnings.
        Default is True.

    Returns
    -------
    final_device_obj : torch.device
        The selected (and validated) PyTorch device object.

    Raises
    ------
    ValueError
        If the device preference is invalid and cannot be resolved.
    RuntimeError
        If the device preference string is malformed or cannot be parsed.

    Examples
    --------
    >>> device = get_pytorch_device() # Auto-detect
    >>> device = get_pytorch_device("cuda:0")
    """
    final_device_obj = None

    # 1. Handle user's explicit device preference
    if device_preference is not None:
        try:
            potential_device = torch.device(device_preference)

            if potential_device.type == 'cuda':
                if not torch.cuda.is_available():
                    if verbose:
                        warnings.warn(
                            f"CUDA preference '{device_preference}' ignored as CUDA is not available. "
                            "Falling back to CPU."
                        )
                    final_device_obj = torch.device("cpu")
                else:
                    # CUDA is available. Validate the specific CUDA device.
                    if potential_device.index is not None:  # e.g., 'cuda:0', 'cuda:1'
                        if potential_device.index >= torch.cuda.device_count():
                            if verbose:
                                warnings.warn(
                                    f"CUDA device index {potential_device.index} in '{device_preference}' is invalid "
                                    f"(found {torch.cuda.device_count()} CUDA devices). "
                                    f"Falling back to default CUDA device 'cuda:0'."
                                )
                            final_device_obj = torch.device("cuda:0")  # Fallback to the first GPU
                        else:
                            final_device_obj = potential_device  # Valid specific CUDA device
                    else:  # User specified 'cuda' (general)
                        final_device_obj = torch.device("cuda")  # Use default/current CUDA device
            
            else:  # User specified a non-CUDA device (e.g., "cpu", "mps")
                final_device_obj = potential_device
        
        except RuntimeError as e:  # Caused by invalid device_preference string like "foo"
            if verbose:
                warnings.warn(
                    f"Invalid device_preference string '{device_preference}': {e}. "
                    "Autodetecting device instead."
                )
            # Let final_device_obj remain None to trigger autodetection below

    # 2. Autodetect if no valid preference was given or processed
    if final_device_obj is None:
        if torch.cuda.is_available():
            final_device_obj = torch.device("cuda:0")  # Default to first CUDA device
            if verbose and device_preference is None: # Only print if it was pure auto-detection
                 print(f"CUDA is available. Defaulting to {torch.cuda.get_device_name(0)} ({str(final_device_obj)}).")
        else:
            final_device_obj = torch.device("cpu")
            if verbose and device_preference is None: # Only warn if they truly didn't specify and are getting CPU
                 warnings.warn("CUDA is not available. Using CPU. This may result in slower performance.")

    # 3. Final confirmation print (if verbose)
    if verbose:
        device_name_str = ""
        if final_device_obj.type == 'cuda':
            try:
                # For 'cuda' without index, get current_device; else, use specified index
                idx = final_device_obj.index if final_device_obj.index is not None else torch.cuda.current_device()
                device_name_str = f" ({torch.cuda.get_device_name(idx)})"
            except Exception: # Should be rare if logic above is correct
                device_name_str = " (Could not fetch GPU name)"
        
        # Avoid printing if it was already covered by specific warnings/prints for fallbacks
        # This condition ensures it prints if the device was chosen without verbose fallbacks
        if not (device_preference is not None and final_device_obj.type == 'cpu' and 'Falling back to CPU' in [w.message.args[0] for w in warnings.catch_warnings(record=True)]): # Crude check
             print(f"Selected device: {str(final_device_obj)}{device_name_str}")
             
    return final_device_obj


def tsne(X, low_dims = 2, perplexity = 30.0, initial_p = 0.5, final_p = 0.8, eta = 500, min_gain = 0.01, T = 1000):
    """
    Computes t-SNE embedding for the input data.
    
    Parameters
    ----------
    X : numpy.narray
        Input data tensor of shape (n_samples, n_features).
    low_dims : int
        Number of dimensions for the output embedding.
    perplexity : float
        Perplexity parameter for t-SNE.
    initial_p : float
        Initial momentum.
    final_p : float
        Final momentum.
    eta : float
        Learning rate.
    min_gain : float
        Minimum gain for gradient updates.
    T : int 
        Number of iterations.
    Returns
    -------
    Y : torch.Tensor
        Low-dimensional representation of the input data.
    kl_div : torch.Tensor
        KL divergence values at each 10 iteration.

    Examples
    --------
        >>> X = np.random.rand(100, 50)  # Example data
        >>> Y = tsne(X, low_dims=2, perplexity=30.0)
        >>> print(Y.shape)  # Should be (100, 2)
    """
    # tolerance for the beta adjustment
    tol = 1e-5
    # get the initial beta
    _, betas = adjustbeta(X, tol, perplexity)

    # Check if CUDA is available
    target_device = get_pytorch_device()

    # change betas into tensor and Move the data to GPU if available
    betas = torch.from_numpy(betas).float().to(target_device)
    X = torch.from_numpy(X).float().to(target_device)

    # Compute pairwise squared distances
    dist_X = compute_pairwise_dist(X)
    # Compute pairwise affinities
    P_joint = compute_pairwise_affinity(dist_X, betas)
    # Normalize, exaggerate and clip P_joint
    # early exaggeration
    early_ex = 4.0
    P_joint = normalize_exaggerate_and_clip(P_joint,early_ex=early_ex)

    # Initialize low-dimensional representation Y and more to GPU
    Y = torch.from_numpy(pca(X.cpu().numpy(), low_dims = low_dims)).to(target_device)
    # Initialize delta_Y and gain
    delta_Y = torch.zeros_like(Y)
    gain = torch.ones_like(Y)
    # 
    kl_div = torch.zeros(T//10,device=target_device)
    for t in tqdm.tqdm(range(T)):
        # Compute pairwise squared distances in low-dimensional space
        dist_Y = compute_pairwise_dist(Y)
        # Compute low-dimensional affinities
        Q = compute_low_dim_affinity(dist_Y)
        # Normalize and clip Q
        Q = torch.clamp(Q, min=1e-12)
        # Compute gradient of the loss function
        dY = compute_gradient_loss_fucntion(P_joint, Q, Y)
        # set the momentum
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
        # get rid of the early exaggeration
        if t == 100:
            P_joint = P_joint / early_ex
            # clip the P_joint
            P_joint = torch.clamp(P_joint, min=1e-12)
        
        # compute kl divergence every 10 iterations
        if t % 10 == 0:
            kl_div_current = compute_kl_divergence(P_joint, Q)
            kl_div[t // 10] = kl_div_current.mean()
            
            
    # print("Q",Q)
    # print("dY",dY)
    return Y,kl_div

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
    X = np.loadtxt('day1/tsne_practice/mnist2500_X.txt')
    print("X shape:", X.shape)
    X = pca(X, 50)
    labels = np.loadtxt('day1/tsne_practice/mnist2500_labels.txt')
    print("labels shape:", labels.shape)
    T = 1000
    Y, kl_div = tsne(X , perplexity = 30.0, eta = 500 , T = T)
    Y = Y.cpu().numpy()
    kl_div = kl_div.cpu().numpy()
    
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.title("t-SNE on MNIST digits")
    plt.savefig("mnist_tsne3.png")

    plt.figure()
    plt.plot(np.arange(0,T,10),kl_div)
    plt.xlabel("Iteration")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence vs Iteration")
    plt.savefig("mnist_kl_div.png")
