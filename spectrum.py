import torch
from torch import nn
# arguments
import yaml
import argparse
from omegaconf import OmegaConf
# model
from src.utils.train_utils import prepare_model
# hessian data
from src.utils.spectrum_utils import prepare_hessian_data
# hessian computation module
from src.lanczos.hessian import hessian
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="experiments/01")
    parser.add_argument("--flag", type=str, default="trained")
    parser.add_argument("--hessian_batch_size", type=int, default=100)
    parser.add_argument("--lanczos", type=str, default="slow")
    parser.add_argument("--iter", type=int, default=100)
    parser_args = parser.parse_args()

    args = OmegaConf.create(yaml.load(open(f"{parser_args.path}/config.yaml"), Loader=yaml.SafeLoader))
    # load model from checkpoint
    model = prepare_model(args)

    if parser_args.flag == "trained":
        model.load_state_dict(torch.load(f"{parser_args.path}/checkpoints/model_trained.pth"))
    elif parser_args.flag == "untrained":
        model.load_state_dict(torch.load(f"{parser_args.path}/checkpoints/model_untrained.pth"))
    else:
        raise ValueError("flag should be either 'trained' or 'untrained'")
    
    # create hessian data loader 
    hessian_dataloader = prepare_hessian_data(args.dataset, parser_args.hessian_batch_size)

    # device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    inputs, targets = next(iter(hessian_dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    # create hessian computation module
    hessian_computation_module = hessian(model, criterion=nn.CrossEntropyLoss(), data=(inputs, targets), cuda=cuda)

    # run slow lanczos algorithm
    eigenvalues_slow, weights_slow = hessian_computation_module.slow_lanczos(iter = parser_args.iter)
          
    # Create directories if they don't exist
    save_dir = f"{parser_args.path}/results/{parser_args.flag}/{parser_args.lanczos}/HBS_{parser_args.hessian_batch_size}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save eigenvalues and weights
    torch.save(eigenvalues_slow, f"{save_dir}/eigenvalues_iter_{parser_args.iter}.pt")
    torch.save(weights_slow, f"{save_dir}/weights_iter_{parser_args.iter}.pt")

if __name__ == "__main__":
    main()