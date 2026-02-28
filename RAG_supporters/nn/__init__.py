"""Neural network models package."""

from RAG_supporters.nn.models import (
    ConfigurableModel,
    JASPERPredictor,
    JASPERPredictorConfig,
    EMAEncoder,
    SubspaceRouter,
    SubspaceRouterConfig,
    DecomposedJASPERPredictor,
    DecomposedJASPERConfig,
)
from RAG_supporters.nn.losses import (
    JASPERLoss,
    ContrastiveLoss,
    CentroidLoss,
    VICRegLoss,
    JASPERMultiObjectiveLoss,
    RoutingLoss,
    EntropyRegularization,
    ResidualPenalty,
    DisentanglementLoss,
)
from RAG_supporters.nn.training import JASPERTrainer, JASPERTrainerConfig, TrainingMonitor
from RAG_supporters.nn.inference import XAIInterface

__all__ = [
    # Phase 1 — Models
    "ConfigurableModel",
    "JASPERPredictor",
    "JASPERPredictorConfig",
    "EMAEncoder",
    # Phase 2 — Models
    "SubspaceRouter",
    "SubspaceRouterConfig",
    "DecomposedJASPERPredictor",
    "DecomposedJASPERConfig",
    # Phase 1 — Losses
    "JASPERLoss",
    "ContrastiveLoss",
    "CentroidLoss",
    "VICRegLoss",
    "JASPERMultiObjectiveLoss",
    # Phase 2 — Losses
    "RoutingLoss",
    "EntropyRegularization",
    "ResidualPenalty",
    "DisentanglementLoss",
    # Training
    "JASPERTrainer",
    "JASPERTrainerConfig",
    "TrainingMonitor",
    # Inference / XAI
    "XAIInterface",
]
