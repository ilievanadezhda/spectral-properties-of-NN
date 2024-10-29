import torch
# arguments
import yaml
import argparse
from omegaconf import OmegaConf
import os

# training utils
from src.utils.train_utils import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="experiments/test")
    parser_args = parser.parse_args()

    args = OmegaConf.create(yaml.load(open(f"{parser_args.path}/config.yaml"), Loader=yaml.SafeLoader))
    # load data
    train_dataloader, test_dataloader = prepare_data(args)
    
    # create model
    model = prepare_model(args)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    # write number of parameters to file
    with open(f"{parser_args.path}/num_parameters.txt", "w") as f:
        f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # create loss function
    criterion = nn.CrossEntropyLoss()
    
    # create optimizer
    optimizer = prepare_optimizer(model, args)
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # create checkpoints directory if it doesn't exist
    os.makedirs(f"{parser_args.path}/checkpoints", exist_ok=True)
    
    # save model before training
    torch.save(model.state_dict(), f"{parser_args.path}/checkpoints/model_untrained.pth")
    
    # train model
    model = train(model, train_dataloader, test_dataloader, criterion, optimizer, args.epochs, device)
    
    # save model after training
    torch.save(model.state_dict(), f"{parser_args.path}/checkpoints/model_trained.pth")

if __name__ == "__main__":
    main()