import torch
import torch.nn as nn

# data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# models
from src.models.NeuralNetwork import NeuralNetwork


def prepare_data(args):
    """Prepare data for training.

    Args:
        args : arguments from config dictionary

    Returns:
        train_dataloader : DataLoader containing training data
        test_dataloader : DataLoader containing test data
    """    
    if args.dataset == "MNIST":
        train_data = datasets.MNIST(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )
        test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=transforms.ToTensor()
        )
        print("Loaded MNIST dataset.")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def prepare_model(args):
    """Prepare model for training.

    Args:
        args : arguments from config dictionary

    Returns:
        model : model for training
    """
    if args.model_name == "NeuralNetwork":
        if args.activation == "Sigmoid":
            model = NeuralNetwork(args.layer_sizes, activation=nn.Sigmoid)
        elif args.activation == "ReLU":
            model = NeuralNetwork(args.layer_sizes, activation=nn.ReLU)
        print(
            f"Initializing NeuralNetwork model with layer sizes {args.layer_sizes} and {args.activation} activation."
        )
    else:
        raise ValueError(f"Model {args.model_name} not supported")
    return model


def prepare_optimizer(model, args):
    """Prepare optimizer for training.

    Args:
        model : model for training
        args : arguments from config dictionary

    Returns:
        optimizer : optimizer for training
    """
    if args.optim_name == "sgd":
        print(
            f"Initializing SGD optimizer with lr={args.optim_lr}, momentum={args.optim_momentum}."
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.optim_momentum
        )
    elif args.optim_name == "adam":
        print(f"Initializing Adam optimizer with lr={args.optim_lr}.")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr)
    return optimizer


def train(
    model,
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    epochs=10,
    device="cpu",
    save_path=None,
):
    """
    Train a neural network model and evaluate on test data

    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader containing training data
        test_dataloader: DataLoader containing test data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        epochs: Number of training epochs (default: 10)
        device: Device to train on (default: "cpu")
    """
    model.to(device)

    for epoch in range(epochs):
        # training phase
        model.train()
        running_train_loss = 0.0
        for batch, (X, y) in enumerate(train_dataloader):
            # move data to device
            X, y = X.to(device), y.to(device)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            pred = model(X)
            loss = criterion(pred, y)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if batch % 100 == 0:
                loss_avg = running_train_loss / (batch + 1)
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch}], Training Loss: {loss_avg:.4f}"
                )

        train_epoch_loss = running_train_loss / len(train_dataloader)

        # testing phase
        model.eval()
        running_test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                running_test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        test_epoch_loss = running_test_loss / len(test_dataloader)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{epochs}] complete:")
        print(f"Training Loss: {train_epoch_loss:.4f}")
        print(f"Test Loss: {test_epoch_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("-" * 60)

        # save model checkpoint
        if (epoch+1) % 5 == 0 and (epoch+1) != epochs and save_path is not None: 
            print(f"Saving model checkpoint at epoch {epoch+1}")
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch+1}.pth")

    print("Training complete!")
    return model
