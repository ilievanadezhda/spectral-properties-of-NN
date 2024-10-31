import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum_histogram(eigenvalues, num_bins, density=False):
    """
    plot the eigenvalue spectrum as a histogram (log scale)
    """
    plt.hist(eigenvalues, bins=num_bins, alpha=0.7, density=density)
    plt.yscale("log")
    plt.xlabel("Eigenvalues", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Eigenvalue Spectrum (Histogram)", fontsize=14)
    plt.grid(True)
    plt.show()


def plot_spectrum(eigenvalues):
    """
    plot the eigenvalue spectrum
    """
    sorted_eigenvalues = np.sort(eigenvalues)
    x_indices = np.arange(1, len(sorted_eigenvalues) + 1)
    plt.plot(x_indices, sorted_eigenvalues, marker="o", linestyle="-", color="red")
    plt.xlabel(r"Index, $i$", fontsize=12)
    plt.ylabel(r"Eigenvalue, $\lambda_{i}$", fontsize=12)
    plt.title("Eigenvalue Spectrum", fontsize=14)
    plt.grid(True)
    plt.show()


def plot_spectrum_combined(eigenvalues, num_bins, title, legend, density=False, path=None):
    """
    plot the eigenvalue spectrum as a histogram and line plot
    """
    # plot histogram
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(title, fontsize=16)
    axs[0].hist(eigenvalues, bins=num_bins, alpha=0.7, density=density)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Eigenvalues", fontsize=12)
    axs[0].set_ylabel("Count", fontsize=12)
    axs[0].set_title("Eigenvalue Spectrum (Histogram)", fontsize=14)
    axs[0].legend([legend])
    axs[0].grid(True)
    # plot line plot
    sorted_eigenvalues = np.sort(eigenvalues)
    x_indices = np.arange(1, len(sorted_eigenvalues) + 1)
    axs[1].plot(x_indices, sorted_eigenvalues, marker="o", linestyle="-", color="red")
    axs[1].set_xlabel(r"Index, $i$", fontsize=12)
    axs[1].set_ylabel(r"Eigenvalue, $\lambda_{i}$", fontsize=12)
    axs[1].set_title("Eigenvalue Spectrum", fontsize=14)
    axs[1].grid(True)
    # plt.figtext(0.5, 0.01, figtext, wrap=False, horizontalalignment="center", fontsize=12)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
