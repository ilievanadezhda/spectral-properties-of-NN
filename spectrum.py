import torch
from torch import nn
# arguments
import yaml
import argparse
from omegaconf import OmegaConf
# time
import time
# model
from src.utils.train_utils import prepare_model
# hessian data
from src.utils.spectrum_utils import prepare_hessian_data
# hessian computation module
from src.lanczos.hessian import hessian
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="experiments/test")
    parser.add_argument("--flag", type=str, default="trained")
    parser.add_argument("--hessian_batch_size", type=int, default=100)
    parser.add_argument("--lanczos", type=str, default="slow")
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=3)
    parser_args = parser.parse_args()

    args = OmegaConf.create(yaml.load(open(f"{parser_args.path}/config.yaml"), Loader=yaml.SafeLoader))

    # create hessian data loader 
    hessian_dataloader = prepare_hessian_data(args.dataset, parser_args.hessian_batch_size)

    # device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # create base directory
    save_dir = f"{parser_args.path}/results/{parser_args.flag}/{parser_args.lanczos}/HBS_{parser_args.hessian_batch_size}"
    os.makedirs(save_dir, exist_ok=True)

    # run 3 times
    for run in range(1, parser_args.num_runs+1):
        # load model from checkpoint
        model = prepare_model(args)
        model.load_state_dict(torch.load(f"{parser_args.path}/checkpoints/model_{parser_args.flag}.pth"))
        model.eval()

        # get new batch of data
        inputs, targets = next(iter(hessian_dataloader))
        inputs, targets = inputs.to(device), targets.to(device)

        # create hessian computation module
        hessian_computation_module = hessian(model, criterion=nn.CrossEntropyLoss(), data=(inputs, targets), cuda=cuda)

        # compute gradient norm
        gradient_norm = hessian_computation_module.get_gradient_norm()

        # run lanczos algorithm
        if parser_args.lanczos == "fast":
            start_time = time.time()
            eigenvalues, weights = hessian_computation_module.fast_lanczos(iter = parser_args.iter)
            end_time = time.time()
        elif parser_args.lanczos == "slow":
            start_time = time.time()
            eigenvalues, weights = hessian_computation_module.slow_lanczos(iter = parser_args.iter)
            end_time = time.time()

        # calculate runtime and print
        runtime = end_time - start_time
        print(f"Runtime: {runtime:.3f} seconds")

        # create run-specific directory
        run_dir = os.path.join(save_dir, f"{run:02d}_run")
        os.makedirs(run_dir, exist_ok=True)
        
        # save runtime, gradient norm, eigenvalues and weights for this run
        torch.save(runtime, f"{run_dir}/runtime_{parser_args.iter}.npy")
        torch.save(gradient_norm, f"{run_dir}/gradient_norm_{parser_args.iter}.npy")
        torch.save(eigenvalues, f"{run_dir}/eigenvalues_iter_{parser_args.iter}.pt")
        torch.save(weights, f"{run_dir}/weights_iter_{parser_args.iter}.pt")

        # delete model and hessian_computation_module
        del model
        del hessian_computation_module
        del inputs
        del targets
        del eigenvalues
        del weights
if __name__ == "__main__":
    main()