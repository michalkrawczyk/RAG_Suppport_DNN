"""Embedding augmentation utilities for data augmentation."""

import random
from typing import Union

import numpy as np
import torch


def random_noise_embedding(
    embedding: Union[np.ndarray, torch.Tensor],
    noise_level: float = 0.01,
    probability: float = 0.5,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Add random Gaussian noise to the embedding.

    Args:
        embedding (np.ndarray): The original embedding vector.
        noise_level (float): The standard deviation of the Gaussian noise to be added.
        probability (float): The probability of applying the noise augmentation.

    Returns:
        np.ndarray: The augmented embedding with added noise.
    """
    if random.random() > probability:
        return embedding
    noise = (
        np.random.normal(0, noise_level, embedding.shape)
        if isinstance(embedding, np.ndarray)
        else torch.normal(0, noise_level, embedding.size())
    )
    augmented_embedding = embedding + noise

    return augmented_embedding


def random_zero_embedding(
    embedding: Union[np.ndarray, torch.Tensor], probability: float = 0.5
) -> Union[np.ndarray, torch.Tensor]:
    """
    Randomly set whole embedding as zero with a given probability.

    This would be used for steering embeddings to 'no information' state.

    Args:
        embedding (np.ndarray or torch.Tensor): The original embedding vector.
        probability (float): The probability of applying the zeroing augmentation.

    Returns:
        np.ndarray or torch.Tensor: The augmented embedding with possible zeroing.
    """
    if random.random() > probability:
        return embedding

    if isinstance(embedding, np.ndarray):
        return np.zeros_like(embedding)
    else:
        return torch.zeros_like(embedding)
