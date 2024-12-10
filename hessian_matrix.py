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
    parser.add_argument("--path", type=str, default="experiments/test")
    parser.add_argument("--hessian_batch_size", type=int, default=60000)
    parser_args = parser.parse_args()

    args = OmegaConf.create(yaml.load(open(f"{parser_args.path}/config.yaml"), Loader=yaml.SafeLoader))

    # create hessian data loader 
    hessian_dataloader = prepare_hessian_data(args.dataset, parser_args.hessian_batch_size)
    
    # device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # prepare inputs and targets
    inputs, targets = next(iter(hessian_dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    # create base directory
    save_dir = f"{parser_args.path}/groundtruth"
    os.makedirs(save_dir, exist_ok=True)

    for checkpoint in os.listdir(f"{parser_args.path}/checkpoints"):
        # skip checkpoints
        if checkpoint not in ["model_untrained.pth", "model_trained.pth"]:
            continue     
        # create checkpoint directory
        checkpoint_dir = os.path.join(save_dir, checkpoint.split('.')[0])
        os.makedirs(checkpoint_dir, exist_ok=True)

        # check if files already exist
        if os.path.exists(f"{checkpoint_dir}/hessian_matrix.pt") and os.path.exists(f"{checkpoint_dir}/eigenvalues.pt"):
            print(f"Files already exist for {checkpoint}")
            continue
        elif os.path.exists(f"{checkpoint_dir}/hessian_matrix.pt"):
            print(f"Hessian matrix already exists for {checkpoint}")
            print("Computing eigendecomposition")

            # load hessian matrix
            hessian_mtx = torch.load(f"{checkpoint_dir}/hessian_matrix.pt")

            # compute eigendecomposition of hessian matrix
            eigenvalues, _ = torch.linalg.eigh(hessian_mtx)
            torch.save(eigenvalues, f"{checkpoint_dir}/eigenvalues.pt")

            # delete hessian matrix and eigenvalues
            del hessian_mtx
            del eigenvalues
        else:
            print(f"Computing hessian matrix and eigendecomposition for {checkpoint}")
            # load model from checkpoint
            model = prepare_model(args)
            model.load_state_dict(torch.load(f"{parser_args.path}/checkpoints/{checkpoint}"))
            model.eval()

            # create hessian computation module
            hessian_computation_module = hessian(model, criterion=nn.CrossEntropyLoss(), data=(inputs, targets), cuda=cuda)

            # compute gradient norm
            print(f"Computing hessian matrix for {checkpoint}")
            hessian_mtx = hessian_computation_module.get_hessian()

            # save hessian matrix
            print(f"Saving hessian matrix for {checkpoint}")
            torch.save(hessian_mtx, f"{checkpoint_dir}/hessian_matrix.pt")

            # delete model and hessian computation module
            del model
            del hessian_computation_module

            # compute eigendecomposition of hessian matrix
            print(f"Computing eigendecomposition for {checkpoint}")
            eigenvalues, _ = torch.linalg.eigh(hessian_mtx)
        
            # save eigenvalues
            print(f"Saving eigenvalues for {checkpoint}")
            torch.save(eigenvalues, f"{checkpoint_dir}/eigenvalues.pt")

            # delete hessian matrix and eigenvalues
            del hessian_mtx
            del eigenvalues

if __name__ == "__main__":
    main()
