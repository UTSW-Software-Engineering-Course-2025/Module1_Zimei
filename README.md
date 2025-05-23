# Dimensionality Reduction Implementations: t-SNE and GraphDR

This project provides Python implementations of two dimensionality reduction techniques: t-distributed Stochastic Neighbor Embedding (t-SNE) and a graph-based dimensionality reduction method (GraphDR). It also includes helper scripts, example data, and Jupyter notebooks for demonstration and testing.

## Project Structure
```bash
.
├── day1/
│   ├── tsne_practice/
│   │   ├── adjustbeta.py         # Helper script for t-SNE beta adjustment
│   │   ├── mnist2500.zip         # MNIST dataset example (extract to get data files)
│   │   ├── t-sne pseudocode.pdf  # Pseudocode for t-SNE
│   │   ├── tsne.ipynb            # Jupyter notebook for t-SNE practice
│   │   └── tsne.py               # Main t-SNE implementation
│   └── Day1_slides.pdf           # Presentation slides for Day 1
├── day3/
│   ├── GraphDR.py              # Main GraphDR implementation
│   ├── GraphDR_test.ipynb      # Jupyter notebook for testing GraphDR
│   ├── Day 3.pdf               # Presentation slides for Day 3
│   ├── hochgerner_2018.anno    # Annotation data
│   └── hochgerner_2018.data.gz # Data file
└── README.md                   # This file



## File Descriptions

### Core Implementations

* **`day1/tsne_practice/tsne.py`**:
    * Contains the main implementation of the t-SNE algorithm.
    * Functions for computing pairwise distances, affinities (both high-dimensional and low-dimensional), KL divergence, and gradients.
    * Includes PCA as a preliminary dimensionality reduction step.
    * Dynamically selects PyTorch device (CPU/GPU).
    * Example usage with the MNIST dataset is provided in the `if __name__ == "__main__":` block.

* **`day3/GraphDR.py`**:
    * Implements a graph-based dimensionality reduction technique (GraphDR).
    * Constructs a k-nearest neighbors graph and uses its Laplacian for dimensionality reduction.
    * Optionally applies rotation based on eigenvectors of $X^T L_{inv} X$.
    * Dynamically selects PyTorch device (CPU/GPU) for certain operations.

### Helper Scripts

* **`day1/tsne_practice/adjustbeta.py`**:
    * A helper module for `tsne.py`.
    * Provides functions to compute entropy and probabilities from a distance matrix (`Hbeta`) and to adjust the precision parameter `beta` based on a target perplexity (`adjustbeta`).

### Jupyter Notebooks

* **`day1/tsne_practice/tsne.ipynb`**:
    * Likely contains experiments, visualizations, and interactive testing related to the `tsne.py` implementation.
* **`day3/GraphDR_test.ipynb`**:
    * Likely used for testing and demonstrating the `GraphDR.py` implementation.

### Data Files

* **`day1/tsne_practice/mnist2500.zip`**:
    * A ZIP archive presumably containing the MNIST dataset subset used in `tsne.py` (e.g., `mnist2500_X.txt` and `mnist2500_labels.txt`).
* **`day3/hochgerner_2018.anno`**:
    * Annotation file, potentially related to the `hochgerner_2018.data.gz` dataset.
* **`day3/hochgerner_2018.data.gz`**:
    * A gzipped data file, likely used for testing or demonstrating GraphDR.

### Supporting Documents

* **`day1/tsne_practice/t-sne pseudocode.pdf`**:
    * A PDF document containing the pseudocode for the t-SNE algorithm.
* **`day1/Day1_slides.pdf`**:
    * Presentation slides, likely covering topics related to t-SNE.
* **`day3/Day 3.pdf`**:
    * Presentation slides for "Day 3", possibly covering GraphDR or related concepts.

## Key Functionalities

### t-SNE (`tsne.py`)

* **`compute_pairwise_dist(X)`**: Calculates pairwise squared Euclidean distances.
* **`adjustbeta(X, tol, perplexity)` (from `adjustbeta.py`)**: Adjusts precision (beta) for each data point based on perplexity.
* **`compute_pairwise_affinity(dist_sq, betas)`**: Computes conditional and joint probabilities ($P_{ij}$) in high-dimensional space.
* **`normalize_exaggerate_and_clip(P, min_clip, early_ex)`**: Normalizes, applies early exaggeration, and clips joint probabilities.
* **`compute_low_dim_affinity(dist_Y)`**: Computes affinities ($Q_{ij}$) in the low-dimensional space using a t-distribution.
* **`compute_kl_divergence(P, Q)`**: Calculates the Kullback-Leibler divergence between P and Q.
* **`compute_gradient_loss_fucntion(P, Q, Y)`**: Computes the gradient of the KL divergence loss function.
* **`tsne(X, low_dims, perplexity, ...)`**: Main function to perform t-SNE embedding.
* **`pca(X, low_dims)`**: Performs PCA for initial dimensionality reduction.

### GraphDR (`GraphDR.py`)

* **`graphdr(data, lambda_, no_rotation=False)`**:
    * Constructs a k-NN graph from the input `data`.
    * Computes the graph Laplacian.
    * Solves a linear system $(I + \lambda L)Z' = X$ to get an initial embedding $Z'$.
    * If `no_rotation` is False, it further refines $Z'$ by projecting it onto the top $d$ eigenvectors of $X^T Z'$, where $d$ is chosen to capture a significant portion (e.g., 95%) of the variance.
* **`get_top_d_eigenvectors(M)`**: Helper function to compute and select eigenvectors.

## Requirements

* Python 3.11
* NumPy
* PyTorch
* scikit-learn (for `kneighbors_graph` in `GraphDR.py`)
* SciPy (for sparse matrix operations and solver in `GraphDR.py`)
* Matplotlib (for plotting examples in `tsne.py`)
* tqdm (for progress bars in `tsne.py`)

## Usage

### t-SNE

The `tsne.py` script can be run directly if the MNIST data (`mnist2500_X.txt` and `mnist2500_labels.txt`) is placed in the `day1/tsne_practice/` directory (after extracting `mnist2500.zip`).

```bash
python day1/tsne_practice/tsne.py
