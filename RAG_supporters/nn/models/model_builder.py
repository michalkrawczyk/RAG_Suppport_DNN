"""Neural network model builder utilities."""

import torch
import torch.nn as nn
import yaml
import numpy as np
import os
import logging
from collections import OrderedDict
from typing import Dict, Any, Optional, Union
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _cast_name_to_class(name: str) -> type:
    """
    Convert string name to PyTorch layer class.

    Args:
        name: String name of the layer class

    Returns:
        PyTorch layer class

    Raises:
        ValueError: If layer name is not found
    """
    custom_layers = {
        # TODO: consider plugin architecture for this (from dnn.layers)
        # "KANLayer": KANLayer
    }

    if name in custom_layers:
        return custom_layers[name]

    if not hasattr(nn, name):
        raise ValueError(f"Layer '{name}' not found in torch.nn nor custom layers")

    return getattr(nn, name)


class ConfigurableModel(nn.Module):
    """
    A PyTorch model that builds itself from a YAML configuration file.

    Attributes:
        input_features: Number of input features
        output_features: Number of output features
        config_path: Path to the configuration file
        device: Device to run the model on
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        device: Optional[torch.device] = None,
        warmup_validate: bool = False,
    ):
        """
        Initialize the configurable model.

        Args:
            config_path: Path to YAML configuration file
            device: Device to run model on (default: auto-detect)
            warmup_validate: Whether to validate model with dummy input
        """
        super().__init__()

        self.config_path = Path(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config file not found at {self.config_path}")

        # Initialize attributes
        self.input_features: int = 0
        self.output_features: int = 0
        self.config: Dict[str, Any] = {}

        # Build model
        self.model = self._build_model()
        self.to(self.device)

        if warmup_validate:
            self._validate_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _load_config(self) -> Dict[str, Any]:
        """
        Load and validate configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            yaml.YAMLError: If YAML parsing fails
            KeyError: If required configuration keys are missing
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)  # Using safe_load for security

            if "model" not in config:
                raise KeyError("Configuration must contain 'model' key")

            if "layers" not in config["model"]:
                raise KeyError("Model configuration must contain 'layers' key")

            return config

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")

    def _build_model(self) -> nn.Sequential:
        """
        Build the neural network model from configuration.

        Returns:
            Sequential model built from configuration
        """
        self.config = self._load_config()
        model_config = self.config["model"]

        model_dict = OrderedDict()
        excluded_keys = {"type", "name"}

        layers_config = model_config["layers"]
        if not layers_config:
            raise ValueError("No layers specified in configuration")

        for i, layer_cfg in enumerate(layers_config):
            # Store input/output features for first/last layers
            if i == 0:
                self.input_features = layer_cfg.get("in_features", 0)
            if i == len(layers_config) - 1:
                self.output_features = layer_cfg.get("out_features", 0)

            layer_name = layer_cfg.get("name", f"{layer_cfg['type']}_{i}")

            try:
                if layer_cfg["type"] == "Sequential":
                    layer = self._build_sequential_layer(layer_cfg, i, excluded_keys)
                else:
                    layer = self._build_single_layer(layer_cfg, excluded_keys)

            except Exception as e:
                LOGGER.error(f"Error building layer '{layer_name}' (index {i}): {e}")
                raise RuntimeError(f"Failed to build layer '{layer_name}': {e}") from e

            model_dict[layer_name] = layer

        return nn.Sequential(model_dict)

    def _build_sequential_layer(
        self, layer_cfg: Dict[str, Any], layer_index: int, excluded_keys: set
    ) -> nn.Sequential:
        """Build a Sequential layer containing multiple sub-layers."""
        layer_dict = OrderedDict()

        for j, sub_layer_cfg in enumerate(layer_cfg["layers"]):
            sub_layer_name = sub_layer_cfg.get("name", f"{sub_layer_cfg['type']}_{layer_index}_{j}")

            try:
                sub_layer = self._build_single_layer(sub_layer_cfg, excluded_keys)
                layer_dict[sub_layer_name] = sub_layer

            except Exception as e:
                raise RuntimeError(
                    f"Error in Sequential layer, sub-layer '{sub_layer_name}': {e}"
                ) from e

        return nn.Sequential(layer_dict)

    def _build_single_layer(self, layer_cfg: Dict[str, Any], excluded_keys: set) -> nn.Module:
        """Build a single layer from configuration."""
        layer_type = layer_cfg["type"]
        layer_args = {k: v for k, v in layer_cfg.items() if k not in excluded_keys}

        layer_class = _cast_name_to_class(layer_type)
        return layer_class(**layer_args)

    def _validate_model(self) -> None:
        """
        Validate the model with a dummy forward pass.

        Raises:
            RuntimeError: If model validation fails
        """
        if self.input_features <= 0:
            LOGGER.warning("Input features not specified, skipping validation")
            return

        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.input_features, device=self.device)

            # Forward pass
            with torch.no_grad():
                output = self.model(dummy_input)

            # Validate output shape
            if self.output_features > 0:
                expected_shape = (1, self.output_features)
                if output.shape != expected_shape:
                    raise RuntimeError(
                        f"Output shape mismatch: expected {expected_shape}, " f"got {output.shape}"
                    )

            LOGGER.info(
                f"Model validation successful. Input: {dummy_input.shape}, Output: {output.shape}"
            )

        except Exception as e:
            raise RuntimeError(f"Model validation failed: {e}") from e

    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.

        Returns:
            String representation of model summary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary = [
            f"Model Configuration: {self.config_path}",
            f"Input Features: {self.input_features}",
            f"Output Features: {self.output_features}",
            f"Device: {self.device}",
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            "\nModel Architecture:",
            str(self.model),
        ]

        return "\n".join(summary)

    def save_config(self, save_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        with open(save_path, "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return f"ConfigurableModel(config='{self.config_path}', device='{self.device}')"
