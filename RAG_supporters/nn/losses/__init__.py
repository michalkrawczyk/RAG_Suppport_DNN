"""JASPER loss functions."""

from RAG_supporters.nn.losses.jasper_losses import (
    CentroidLoss,
    ContrastiveLoss,
    JASPERLoss,
    JASPERMultiObjectiveLoss,
    VICRegLoss,
)
from RAG_supporters.nn.losses.routing_losses import (
    DisentanglementLoss,
    EntropyRegularization,
    ResidualPenalty,
    RoutingLoss,
)

__all__ = [
    # Phase 1
    "JASPERLoss",
    "ContrastiveLoss",
    "CentroidLoss",
    "VICRegLoss",
    "JASPERMultiObjectiveLoss",
    # Phase 2
    "RoutingLoss",
    "EntropyRegularization",
    "ResidualPenalty",
    "DisentanglementLoss",
]
