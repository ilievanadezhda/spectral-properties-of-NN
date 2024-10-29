import argparse
import os
import torch
import matplotlib.pyplot as plt
from src.utils.plot_utils import plot_spectrum_combined

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="experiments/01") 
    parser_args = parser.parse_args()

    # get the results directory path
    results_dir = f"{parser_args.path}/results"
    print(f"Results directory: {results_dir}")
    
    # walk through all subdirectories
    for flag in os.listdir(results_dir):
        flag_dir = os.path.join(results_dir, flag)
        if os.path.isdir(flag_dir):
            for lanczos in os.listdir(flag_dir):
                lanczos_dir = os.path.join(flag_dir, lanczos)
                if os.path.isdir(lanczos_dir):
                    for batch_dir in os.listdir(lanczos_dir):
                        if batch_dir.startswith('HBS_'):
                            # extract batch size from directory name
                            batch_size = int(batch_dir.split('_')[1])
                            batch_path = os.path.join(lanczos_dir, batch_dir)
                            
                            # iterate through run folders
                            for run_dir in os.listdir(batch_path):
                                if run_dir.endswith('_run'):
                                    run_path = os.path.join(batch_path, run_dir)
                                    run_num = int(run_dir.split('_')[0])
                                    
                                    # find iteration files
                                    for file in os.listdir(run_path):
                                        if file.startswith('eigenvalues_iter_'):
                                            # extract number of iterations
                                            iter_num = int(file.split('_')[-1].split('.')[0])
                                            
                                            # load eigenvalues and weights
                                            eigenvalues = torch.load(os.path.join(run_path, f'eigenvalues_iter_{iter_num}.pt'))
                                            weights = torch.load(os.path.join(run_path, f'weights_iter_{iter_num}.pt'))
                                            
                                            # create title for the plot
                                            title = f"Spectrum: {flag.capitalize()} Model"
                                            legend = f"Details:\nAlgorithm: {lanczos.capitalize()} Lanczos\nHessian Batch Size: {batch_size}\nLanczos Iterations: {iter_num}\nRun: {run_num}"
                                            
                                            # save plot
                                            plot_path = os.path.join(run_path, f'spectrum_plot_iter_{iter_num}.png')
                                            plot_spectrum_combined(eigenvalues, num_bins=100, title=title, legend=legend, path=plot_path)
                                
    
if __name__ == "__main__":
    main()