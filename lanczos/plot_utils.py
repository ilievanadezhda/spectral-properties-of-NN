import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum_histogram(eigenvalues, num_bins, density=False):
    """
    plot the eigenvalue spectrum as a histogram (log scale)
    """
    plt.hist(eigenvalues, bins=num_bins, alpha=0.7, density=density)
    plt.yscale('log')
    plt.xlabel('Eigenvalues', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Eigenvalue Spectrum (Histogram)', fontsize=14)
    plt.grid(True)
    plt.show()


def plot_spectrum(eigenvalues):
    """
    plot the eigenvalue spectrum
    """
    sorted_eigenvalues = np.sort(eigenvalues)
    x_indices = np.arange(1, len(sorted_eigenvalues) + 1)
    plt.plot(x_indices, sorted_eigenvalues, marker='o', linestyle='-', color='red')
    plt.xlabel(r'Index, $i$', fontsize=12)
    plt.ylabel(r'Eigenvalue, $\lambda_{i}$', fontsize=12)
    plt.title('Eigenvalue Spectrum', fontsize=14)
    plt.grid(True)
    plt.show()


def plot_spectrum_combined(eigenvalues, num_bins, title, density=False):
    """
    plot the eigenvalue spectrum as a histogram and line plot
    """
    # plot histogram
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    axs[0].hist(eigenvalues, bins=num_bins, alpha=0.7, density=density)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Eigenvalues', fontsize=12)
    axs[0].set_ylabel('Count', fontsize=12)
    axs[0].set_title('Eigenvalue Spectrum (Histogram)', fontsize=14)
    axs[0].grid(True)
    # plot line plot
    sorted_eigenvalues = np.sort(eigenvalues)
    x_indices = np.arange(1, len(sorted_eigenvalues) + 1)
    axs[1].plot(x_indices, sorted_eigenvalues, marker='o', linestyle='-', color='red')
    axs[1].set_xlabel(r'Index, $i$', fontsize=12)
    axs[1].set_ylabel(r'Eigenvalue, $\lambda_{i}$', fontsize=12)
    axs[1].set_title('Eigenvalue Spectrum', fontsize=14)
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()


def get_esd_plot(eigenvalues, weights, title):
    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.grid(True)
    plt.show()


def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :].real)
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)