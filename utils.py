"""Utility methods and constants used throughout the project."""
import os

import torch
from torch import nn
from torchvision import transforms

from faces_dataset import FacesDataset
from models import EfficientnetB0, EfficientnetB0Triplet
from torch.nn.modules.distance import PairwiseDistance


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(256, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


def load_dataset(dataset_path: str, dataset_part: str, triplet: bool):
    """Loads dataset part from dataset path.
    Args:
        dataset_path: explicit dataset_path, it should contains train, val and test folder,
                      and each one of them should contain fake and real folders
        dataset_part: dataset part, one of: train, val, test.
        triplet: Determine whether to load triplet or two images

    Returns:
        dataset: a torch.utils.dataset.Dataset instance.
    """
    transform = {'train': TRANSFORM_TRAIN,
                 'val': TRANSFORM_TEST,
                 'test': TRANSFORM_TEST}[dataset_part]
    dataset = FacesDataset(
        root_path=os.path.join(dataset_path,
                               dataset_part),
        transform=transform, triplet=triplet)
    return dataset


def choose_criterion(model_name: str):
    """Choose loss function based on model (original or triplet version).

    Args:
        model_name: the name of the model, one of: Origin, Triplet.
    Returns:
        criterion: loss to be used for training
    """
    criterion = {
        'Origin': nn.CrossEntropyLoss(),
        'Triplet': torch.nn.TripletMarginLoss(margin=0.2, p=2),
    }

    if model_name not in criterion:
        raise ValueError(f"Invalid Model name {model_name}")

    return criterion[model_name]


def load_model(model_name: str, embedding_dimension: int):
    """Load the model corresponding to the name given.

    Args:
        model_name: the name of the model, one of: Origin, Triplet.
        embedding_dimension: Dimension of the embedding vector
    Returns:
        model: the model initialized, and loaded to device.
    """
    models = {
        'Origin': EfficientnetB0(),
        'Triplet': EfficientnetB0Triplet(embedding_dimension),
    }

    if model_name not in models:
        raise ValueError(f"Invalid Model name {model_name}")

    print(f"Building model {model_name}...")
    model = models[model_name]
    model = model.to(device)
    return model


def get_nof_params(model: nn.Module) -> int:
    """Return the number of trainable model parameters.

    Args:
        model: nn.Module.

    Returns:
        The number of model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
