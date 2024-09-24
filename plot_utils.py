import numpy as np
import matplotlib.pyplot as plt


def plot_hessian_spectrum_hist(eigenvalues, num_bins=500):
    """
    plot the eigenvalues of the Hessian matrix as a histogram (log scale)
    """
    plt.hist(eigenvalues, bins=num_bins, color='b', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Eigenvalues', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Hessian Spectrum', fontsize=14)
    plt.grid(True)
    plt.show()


def plot_hessian_spectrum_density(eigenvalues, num_bins=500):
    """
    plot the density of the eigenvalues of the Hessian matrix
    """
    plt.hist(eigenvalues, bins=num_bins, density=True, color='b', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Eigenvalues', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Hessian Eigenvalues Spectral Density (ESD)', fontsize=14)
    plt.grid(True)
    plt.show()


def plot_hessian_spectrum_density_(eigenvalues, num_bins=500):
    """
    Plot the density of the eigenvalues of the Hessian matrX
    """
    counts, bin_edges = np.histogram(eigenvalues, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, counts, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('Eigenvalue', fontsize=12)
    plt.ylabel('Density (log scale)', fontsize=12)
    plt.title('Hessian Eigenvalues Spectral Density (ESD)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def plot_hessian_spectrum(eigenvalues):
    """
    plot the eigenvalues of the Hessian matrix
    """
    sorted_eigenvalues = np.sort(eigenvalues)
    x_indices = np.arange(1, len(sorted_eigenvalues) + 1)
    plt.plot(x_indices, sorted_eigenvalues, marker='o', linestyle='-', color='red')
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title('Hessian Spectrum', fontsize=14)
    plt.grid(True)
    plt.show()