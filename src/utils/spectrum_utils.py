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
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    print(f"Batch size for Hessian computation: {hessian_batch_size}")
    return hessian_dataloader
