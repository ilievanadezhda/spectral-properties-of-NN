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


def plot_spectrum_combined(eigenvalues, num_bins, title, density=False):
    """
    plot the eigenvalue spectrum as a histogram and line plot
    """
    # plot histogram
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    axs[0].hist(eigenvalues, bins=num_bins, alpha=0.7, density=density)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Eigenvalues", fontsize=12)
    axs[0].set_ylabel("Count", fontsize=12)
    axs[0].set_title("Eigenvalue Spectrum (Histogram)", fontsize=14)
    axs[0].grid(True)
    # plot line plot
    sorted_eigenvalues = np.sort(eigenvalues)
    x_indices = np.arange(1, len(sorted_eigenvalues) + 1)
    axs[1].plot(x_indices, sorted_eigenvalues, marker="o", linestyle="-", color="red")
    axs[1].set_xlabel(r"Index, $i$", fontsize=12)
    axs[1].set_ylabel(r"Eigenvalue, $\lambda_{i}$", fontsize=12)
    axs[1].set_title("Eigenvalue Spectrum", fontsize=14)
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()


def plot_lanczos_iterations(data, title):
    """
    plot the eigenvalues of each iteration of the Lanczos algorithm (+ groundtruth)
    Fig 7.2. in Applied Numerical Linear Algebra
    """
    x_positions = [i for i in range(1, len(data)+1)]
    plt.figure(figsize=(12, 6))
    for i, values in enumerate(data[:-1]):
        values = np.sort(np.array(values))
        x_values = np.full_like(values, x_positions[i])
        for j in range(len(values)):
            if j == 0 or j == len(values) - 1:
                plt.scatter(x_values[j], values[j], color="black", marker="x")
            elif j == 1 or j == len(values) - 2:
                plt.scatter(x_values[j], values[j], color="red", marker="x")
            elif j == 2 or j == len(values) - 3:
                plt.scatter(x_values[j], values[j], color="green", marker="x")
            elif j == 3 or j == len(values) - 4:
                plt.scatter(x_values[j], values[j], color="blue", marker="x")
            elif j == 4 or j == len(values) - 5:
                plt.scatter(x_values[j], values[j], color="orange", marker="x")
            else:
                plt.scatter(x_values[j], values[j], color="gray", marker="x")
    # plot the groundtruth
    values = np.sort(np.array(data[-1]))
    x_values = np.full_like(values, x_positions[-1])
    plt.scatter(x_values, values, color="black", marker="x")
    # set x-ticks and labels
    # if the number of iterations is large, only show every 10th iteration starting from 0
    if len(data) > 25:
        x_positions = [i for i in range(0, len(data)+1, 10)]
        x_positions[0] = 1
    plt.xticks(x_positions, [f"{i}" for i in x_positions[:-1]] + ["G"])
    plt.xlabel("Iteration")
    plt.ylabel("Eigenvalues")
    plt.title(f"Eigenvalues of {title} iterations")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.show()


def get_esd_plot(eigenvalues, weights):
    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
    plt.title('Hessian Eigenvalues Spectral Density', fontsize=14)
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