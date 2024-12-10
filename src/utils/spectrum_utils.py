from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def prepare_hessian_data(dataset, hessian_batch_size):
    """Prepare data for Hessian computation.

    Args:
        args : arguments from config dictionary

    Returns:
        hessian_dataloader : DataLoader containing data for Hessian computation
    """

    if dataset == "MNIST":
        training_data = datasets.MNIST(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )
        if hessian_batch_size > len(training_data):
            raise ValueError(
                f"Batch size {hessian_batch_size} is larger than the dataset size {len(training_data)}"
            )
        hessian_dataloader = DataLoader(
            training_data,
            batch_size=hessian_batch_size,
            shuffle=True,
        )
        print("Loaded MNIST dataset for Hessian computation.")
        if hessian_batch_size == len(training_data):
            print("Using full dataset for Hessian computation.")
    elif dataset == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        training_data = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform
        )
        if hessian_batch_size > len(training_data):
            raise ValueError(
                f"Batch size {hessian_batch_size} is larger than the dataset size {len(training_data)}"
            )
        hessian_dataloader = DataLoader(
            training_data,
            batch_size=hessian_batch_size,
            shuffle=True,
        )
        print("Loaded CIFAR10 dataset for Hessian computation.")
        if hessian_batch_size == len(training_data):
            print("Using full dataset for Hessian computation.")
    elif dataset == "CIFAR10_grayscale":
        # transform CIFAR10 images to grayscale and resize to 28x28
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        )
        training_data = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform
        )
        if hessian_batch_size > len(training_data):
            raise ValueError(
                f"Batch size {hessian_batch_size} is larger than the dataset size {len(training_data)}"
            )
        hessian_dataloader = DataLoader(
            training_data,
            batch_size=hessian_batch_size,
            shuffle=True,
        )
        print("Loaded CIFAR10 dataset with grayscale images for Hessian computation.")
        if hessian_batch_size == len(training_data):
            print("Using full dataset for Hessian computation.")
    elif dataset == "FashionMNIST":
        training_data = datasets.FashionMNIST(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )
        if hessian_batch_size > len(training_data):
            raise ValueError(
                f"Batch size {hessian_batch_size} is larger than the dataset size {len(training_data)}"
            )
        hessian_dataloader = DataLoader(
            training_data,
            batch_size=hessian_batch_size,
            shuffle=True,
        )
        print("Loaded FashionMNIST dataset for Hessian computation.")
        if hessian_batch_size == len(training_data):
            print("Using full dataset for Hessian computation.")
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    print(f"Batch size for Hessian computation: {hessian_batch_size}")
    return hessian_dataloader
