"""What's In My Human Feedback (WIMHF) package."""

from .quickstart import load_config, run_wimhf_pipeline, WIMHFConfig
from .sae import SparseAutoencoder, get_sae_checkpoint_name, load_model
from .embedding import get_openai_embeddings, get_local_embeddings
from .interpretation import (
    NeuronInterpreter,
    InterpretConfig,
    SamplingConfig,
    ScoringConfig,
)
from .feature_selection import (
    select_neurons_controlled_lasso,
    select_neurons_controlled_ols,
    select_neurons_lasso,
)

__all__ = [
    "SparseAutoencoder",
    "get_sae_checkpoint_name",
    "load_model",
    "get_openai_embeddings",
    "get_local_embeddings",
    "NeuronInterpreter",
    "InterpretConfig",
    "SamplingConfig",
    "ScoringConfig",
    "select_neurons_controlled_lasso",
    "select_neurons_controlled_ols",
    "select_neurons_lasso",
    "run_wimhf_pipeline",
    "load_config",
    "WIMHFConfig",
]

__version__ = "0.1.0"
