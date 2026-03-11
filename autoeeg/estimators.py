import numpy as np
from autoeeg.base_estimator import BaseEEGEstimator
from mindware.components.utils.constants import type_dict
from sklearn.utils.multiclass import type_of_target
from mindware.components.feature_engineering.transformation_graph import DataNode

class EEGClassifier(BaseEEGEstimator):
    def initialize(self, data: DataNode, **kwargs):
        print("[DEBUG] EEGClassifier.initialize() started")
        
        # 1. Robust Task type detection
        # EEGDataNode.data[1] can be List[events] or 1D Labels array
        y = data.data[1]
        if isinstance(y, list):
            # Extract 3rd column (labels) from each event array and concatenate
            all_labels = np.concatenate([evt[:, 2] for evt in y])
        elif isinstance(y, np.ndarray) and len(y.shape) == 2 and y.shape[1] == 3:
            # Single MNE event object
            all_labels = y[:, 2]
        else:
            # Already a 1D labels array
            all_labels = y

        task_type = type_of_target(all_labels)
        if task_type in type_dict:
            self.task_type = type_dict[task_type]
        else:
            raise ValueError("Invalid Task Type: %s!" % task_type)
        
        print(f"[DEBUG] Task Type detected: {task_type} -> {self.task_type}")
        
        # 2. Call super initialize
        super().initialize(data, **kwargs)
        print("[DEBUG] EEGClassifier.initialize() completed")

    def fit(self, data: DataNode, **kwargs):
        print("[DEBUG] EEGClassifier.fit() called")
        if self._ml_engine is None:
            self.initialize(data, **kwargs)
        
        super().fit(data, **kwargs)
        return self
