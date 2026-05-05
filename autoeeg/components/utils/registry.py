from autoeeg.components.feature_engineering.transformations import (
    Resampler, FIRFilter, IIRFilter, CARTransformer, Segmenter, CSPTransformer, Identity
)
from autoeeg.components.models.classification import (
    LDA, SVM, RandomForest, LR, QDA
)

# Registry for Feature Engineering Transformers
TRANSFORMER_REGISTRY = {
    "Resampler": Resampler,
    "FIRFilter": FIRFilter,
    "IIRFilter": IIRFilter,
    "CARTransformer": CARTransformer,
    "Segmenter": Segmenter,
    "CSPTransformer": CSPTransformer,
    "Identity": Identity
}

# Registry for Classification Models
MODEL_REGISTRY = {
    "LDA": LDA,
    "SVM": SVM,
    "RandomForest": RandomForest,
    "LR": LR,
    "QDA": QDA
}

def get_transformer_class(name):
    if name not in TRANSFORMER_REGISTRY:
        raise ValueError(f"Transformer '{name}' not found in registry.")
    return TRANSFORMER_REGISTRY[name]

def get_model_class(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    return MODEL_REGISTRY[name]
