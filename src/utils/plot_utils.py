import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.utils.general_utils import count_correctly_estimated_eigenvalues


def plot_spectrum_histogram(eigenvalues, num_bins, density=False):
    """
    plot the eigenvalue spectrum as a histogram (log scale)
    """
    plt.hist(eigenvalues, bins=num_bins, alpha=0.7, density=density)
    plt.yscale("log")
    plt.xlabel("Eigenvalues", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Eigenvalue Spectrum (Histogram)", fontsize=14)
    plt.grid(True, alpha=0.5, linestyle="--")
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
    plt.grid(True, alpha=0.5, linestyle="--")
    plt.show()


def plot_spectrum_combined(
    eigenvalues, num_bins, title, legend, density=False, path=None
):
    """
    plot the eigenvalue spectrum as a histogram and line plot
    """
    # plot histogram
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(title.replace("_", " "), fontsize=16)
    axs[0].hist(
        eigenvalues, bins=num_bins, alpha=0.7, edgecolor="black", density=density
    )
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Eigenvalues", fontsize=12)
    axs[0].set_ylabel("Count", fontsize=12)
    axs[0].set_title("Eigenvalue Spectrum (Histogram)", fontsize=14)
    axs[0].legend([legend])
    axs[0].grid(True, alpha=0.5, linestyle="--")
    # plot line plot
    sorted_eigenvalues = np.sort(eigenvalues)
    x_indices = np.arange(1, len(sorted_eigenvalues) + 1)
    axs[1].plot(
        x_indices,
        sorted_eigenvalues,
        marker="o",
        linestyle="",
        color="red",
        markersize=3,
    )
    axs[1].axhline(
        y=0, color="black", linestyle="-", linewidth=0.5
    )  # Add horizontal line at y=0
    axs[1].set_xlabel(r"Index, $i$", fontsize=12)
    axs[1].set_ylabel(r"Eigenvalue, $\lambda_{i}$", fontsize=12)
    axs[1].set_title("Eigenvalue Spectrum", fontsize=14)
    axs[1].grid(True, alpha=0.5, linestyle="--")
    # plt.figtext(0.5, 0.01, figtext, wrap=False, horizontalalignment="center", fontsize=12)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_groundtruth_and_lanczos(
    lanczos_eigenvalues, gt_eigenvalues, plot_title, LANCZOS, ITER
):
    """
    plot the comparison of ground-truth eigenvalues and lanczos eigenvalues
    """
    # create figure
    plt.figure(figsize=(18, 6))

    # plot lanczos eigenvalues
    plt.subplot(1, 3, 1)
    sorted_lanczos_eigenvalues = np.sort(lanczos_eigenvalues)
    x_indices_lanczos = np.arange(1, len(sorted_lanczos_eigenvalues) + 1)
    plt.plot(
        x_indices_lanczos,
        sorted_lanczos_eigenvalues,
        marker="o",
        linestyle="",
        color="red",
        markersize=3,
    )
    plt.axhline(
        y=0, color="black", linestyle="-", linewidth=0.5
    )  # add horizontal line at y=0
    plt.title(
        f"{LANCZOS.capitalize()} Lanczos Eigenvalues\nLanczos Iterations: {ITER}",
        fontsize=14,
    )
    plt.xlabel(r"Index, $i$", fontsize=12)
    plt.ylabel(r"Eigenvalue, $\lambda_{i}$", fontsize=12)
    plt.grid(True, alpha=0.5, linestyle="--")

    # plot ground truth eigenvalues
    plt.subplot(1, 3, 2)
    sorted_gt_eigenvalues = np.sort(gt_eigenvalues.numpy())
    x_indices_gt = np.arange(1, len(sorted_gt_eigenvalues) + 1)
    plt.plot(
        x_indices_gt,
        sorted_gt_eigenvalues,
        marker="o",
        linestyle="",
        color="green",
        markersize=3,
        label="Ground Truth",
    )
    plt.axhline(
        y=0, color="black", linestyle="-", linewidth=0.5
    )  # add horizontal line at y=0
    plt.title(f"Ground-truth Eigenvalues", fontsize=14)
    plt.xlabel(r"Index, $i$", fontsize=12)
    plt.ylabel(r"Eigenvalue, $\lambda_{i}$", fontsize=12)
    plt.grid(True, alpha=0.5, linestyle="--")

    # plot first 50 and last 50 ground truth eigenvalues
    plt.subplot(1, 3, 3)
    first_50_gt_eigenvalues = sorted_gt_eigenvalues[:50]
    last_50_gt_eigenvalues = sorted_gt_eigenvalues[-50:]
    combined_gt_eigenvalues = np.concatenate(
        (first_50_gt_eigenvalues, last_50_gt_eigenvalues)
    )
    x_indices_combined = np.arange(1, len(combined_gt_eigenvalues) + 1)
    x_labels = np.concatenate(
        (np.arange(0, 50), np.arange(len(gt_eigenvalues) - 49, len(gt_eigenvalues)))
    )
    plt.plot(
        x_indices_combined,
        combined_gt_eigenvalues,
        marker="o",
        linestyle="",
        color="green",
        markersize=3,
        label="Ground-truth",
    )
    plt.plot(
        x_indices_lanczos,
        sorted_lanczos_eigenvalues,
        marker="o",
        linestyle="",
        color="red",
        markersize=3,
        alpha=0.3,
        label="Lanczos",
    )
    plt.xticks(x_indices_combined[::10], x_labels[::10])
    plt.axhline(
        y=0, color="black", linestyle="-", linewidth=0.5
    )  # add horizontal line at y=0
    plt.title(f"First 50 and Last 50 Ground-truth Eigenvalues", fontsize=14)
    plt.xlabel(r"Index, $i$", fontsize=12)
    plt.ylabel(r"Eigenvalue, $\lambda_{i}$", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5, linestyle="--")
    # plt.suptitle("Comparison of Lanczos Spectrum and Ground-truth Spectrum", fontsize=18)
    plt.suptitle(plot_title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_lanczos_iterations(
    lanczos_eigenvalues, sorted_gt_eigenvalues, LANCZOS, ITERS, HBS, plot_title
):
    """
    plot the spectra for different number of Lanczos iterations
    """
    plt.figure(figsize=(18, 12))
    for i, ITER in enumerate(ITERS):
        plt.subplot(2, 3, i + 1)
        # lanczos eigenvalues
        sorted_lanczos_eigenvalues = np.sort(lanczos_eigenvalues[ITER])
        x_indices_lanczos = np.arange(1, len(sorted_lanczos_eigenvalues) + 1)
        # groundtruth eigenvalues
        x = ITER // 2
        first_x_gt_eigenvalues = sorted_gt_eigenvalues[:x]
        last_x_gt_eigenvalues = sorted_gt_eigenvalues[-x:]
        combined_gt_eigenvalues = np.concatenate(
            (first_x_gt_eigenvalues, last_x_gt_eigenvalues)
        )
        # plot
        plt.plot(
            x_indices_lanczos,
            sorted_lanczos_eigenvalues,
            marker="o",
            linestyle="",
            color="red",
            markersize=3,
            label="Lanczos Eigenvalues",
        )
        plt.plot(
            x_indices_lanczos,
            combined_gt_eigenvalues,
            marker="o",
            linestyle="",
            color="green",
            markersize=3,
            alpha=0.4,
            label=f"First {x} and Last {x} Groundtruth Eigenvalues",
        )
        plt.axhline(
            y=0, color="black", linestyle="-", linewidth=0.5
        )  # add horizontal line at y=0
        plt.title(f"Lanczos Iterations: {ITER}", fontsize=14)
        plt.xlabel(r"Index, $i$", fontsize=10)
        plt.ylabel(r"Eigenvalue, $\lambda_{i}$", fontsize=10)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.5, linestyle="--")

    plt.suptitle(
        f"{plot_title}\nAlgorithm: {LANCZOS.capitalize()} Lanczos", fontsize=14
    )
    plt.tight_layout()
    plt.show()


def plot_spectrum_evolution(all_eigenvalues, configs, HBS, plot_title):
    """
    plot the spectrum along the training trajectory
    """
    # plot the eigenvalues of all configurations
    fig, axes = plt.subplots(
        1, len(all_eigenvalues), figsize=(len(all_eigenvalues) * 4, 6)
    )
    for i, eigenvalues in enumerate(all_eigenvalues):
        if len(all_eigenvalues) > 1:
            ax = axes[i]
        else:
            ax = axes
        sorted_eigenvalues = np.sort(eigenvalues)
        x_indices = np.arange(1, len(sorted_eigenvalues) + 1)
        ax.plot(
            x_indices,
            sorted_eigenvalues,
            marker="o",
            linestyle="",
            color="red",
            markersize=3,
        )
        ax.axhline(
            y=0, color="black", linestyle="-", linewidth=0.5
        )  # Add horizontal line at y=0
        ax.set_title(f'{configs[i]["model_state"].capitalize().replace("_", " ")}')
        legend = [
            Patch(
                facecolor="white",
                edgecolor="white",
                label=rf"$\| \nabla \mathcal{{L}}_{{ {HBS} }} \|_2 = {np.round(configs[i]['gradient_norm'], 3)}$",
            )
        ]
        ax.legend(handles=legend, loc="upper left", handlelength=0, handletextpad=0)
        ax.set_xlabel("Eigenvalue", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, alpha=0.5)
    fig.suptitle(plot_title)
    plt.tight_layout()

    # plot histograms of eigenvalues of all configurations
    fig, axes = plt.subplots(
        1, len(all_eigenvalues), figsize=(len(all_eigenvalues) * 4, 6)
    )
    for i, eigenvalues in enumerate(all_eigenvalues):
        if len(all_eigenvalues) > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.hist(eigenvalues, bins=100, alpha=0.7, edgecolor="black")
        ax.set_title(f'{configs[i]["model_state"].capitalize().replace("_", " ")}')
        legend = [
            Patch(
                facecolor="white",
                edgecolor="white",
                label=rf"$\| \nabla \mathcal{{L}}_{{ {HBS} }} \|_2 = {np.round(configs[i]['gradient_norm'], 3)}$",
            )
        ]
        ax.legend(handles=legend, loc="upper left", handlelength=0, handletextpad=0)
        ax.set_xlabel("Eigenvalue", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_ylim(0, 25)
        ax.grid(True, alpha=0.5)
    fig.suptitle(plot_title)
    plt.tight_layout()
    plt.show()


def plot_number_correct_eigenvalues(
    results_dir, HBS, ALL_ITERS, gt_eigenvalues, plot_title, LANCZOS):
    """
    plot number of lanczos iterations vs the count of correctly estimated eigenvalues
    """
    num_correct_eigenvalues = {}
    for ITER in ALL_ITERS:
        for run in range(1, 11):
            eigenvalues = torch.load(os.path.join(
                results_dir, f"HBS_{HBS}/{run:02d}_run/eigenvalues_iter_{ITER}.pt"))
            if run == 1:
                num_correct_eigenvalues[ITER] = [
                    count_correctly_estimated_eigenvalues(eigenvalues, gt_eigenvalues, ITER)]
            else:
                num_correct_eigenvalues[ITER].append(
                    count_correctly_estimated_eigenvalues(eigenvalues, gt_eigenvalues, ITER))
    
    # calculate means and standard deviations
    means = [np.mean(num_correct_eigenvalues[ITER]) for ITER in ALL_ITERS]
    stds = [np.std(num_correct_eigenvalues[ITER]) for ITER in ALL_ITERS]

    # plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(ALL_ITERS, means, yerr=stds, fmt='-o', capsize=10)
    for i, txt in enumerate(means):
        offset = (0, 10) if txt < max(means) else (0, -15)
        plt.annotate(
            txt,
            (ALL_ITERS[i], means[i]),
            textcoords="offset points",
            xytext=offset,
            ha="center",
        )
    plt.xlabel("Number of Lanczos Iterations")
    plt.ylabel("Number of Correctly Estimated Eigenvalues")
    plt.title(f"{plot_title}\nAlgorithm: {LANCZOS.capitalize()} Lanczos\n"
              "Number of Lanczos Iterations vs Correctly Estimated Eigenvalues")
    plt.grid(True)
    plt.show()


# def plot_number_correct_eigenvalues(
#     lanczos_eigenvalues, gt_eigenvalues, LANCZOS, ITERS, plot_title
# ):
#     """
#     plot number of lanczos iterations vs the count of correctly estimated eigenvalues
#     """
#     correct_eigenvalues = [
#         count_correctly_estimated_eigenvalues(
#             lanczos_eigenvalues[ITER], gt_eigenvalues, ITER
#         )
#         for ITER in ITERS
#     ]
#     plt.figure(figsize=(10, 6))
#     plt.plot(ITERS, correct_eigenvalues, marker="o")
#     for i, txt in enumerate(correct_eigenvalues):
#         offset = (0, 10) if txt < max(correct_eigenvalues) else (0, -15)
#         plt.annotate(
#             txt,
#             (ITERS[i], correct_eigenvalues[i]),
#             textcoords="offset points",
#             xytext=offset,
#             ha="center",
#         )
#     plt.xlabel("Number of Lanczos Iterations")
#     plt.ylabel("Number of Correctly Estimated Eigenvalues")
#     plt.title(
#         f"{plot_title}\nAlgorithm: {LANCZOS.capitalize()} Lanczos\nNumber of Lanczos Iterations vs Correctly Estimated Eigenvalues"
#     )
#     plt.grid(True)
#     plt.show()