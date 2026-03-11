import mne
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from autoeeg.components.feature_engineering.transformations.base_eeg_transformer import EEGTransformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class CSPTransformer(EEGTransformer):
    def __init__(self, n_components=4, reg=None, log=True, random_state=1):
        super().__init__("spatial_filter", random_state)
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.type = 105
        self.model = None

    def _operate(self, data_node: EEGDataNode):
        X, y = data_node.data
        
        # 1. Robust Dimension Check
        if isinstance(X, list) or len(X.shape) != 3:
            raise ValueError(
                f"CSP requires 3D numpy array (n_epochs, n_channels, n_times). "
                f"Current X type: {type(X)}. Did you put 'segmenter' before 'spatial_filter'?"
            )

        # 2. Robust Label Extraction
        # If y is an MNE event object (n_events, 3), take the 3rd column as labels
        if isinstance(y, np.ndarray) and len(y.shape) == 2 and y.shape[1] == 3:
            actual_labels = y[:, 2]
        else:
            actual_labels = y

        if self.model is None:
            # Training Mode
            reg_param = self.reg if self.reg != 'none' else None
            self.model = mne.decoding.CSP(n_components=self.n_components, 
                                          reg=reg_param, 
                                          log=self.log, 
                                          norm_trace=False)
            
            try:
                X_transformed = self.model.fit_transform(X, actual_labels)
            except Exception as e:
                print(f"CSP Fit error: {e}")
                raise e
        else:
            # Test/Val Mode
            X_transformed = self.model.transform(X)

        new_feature_types = ['numeric'] * X_transformed.shape[1]
        
        # Return 2D node. Note: we return 'actual_labels' to ensure consistency 
        # for the next stage (classifier)
        new_node = EEGDataNode(data=[X_transformed, actual_labels], 
                               feature_type=new_feature_types, 
                               task_type=data_node.task_type, 
                               subject_ids=data_node.subject_ids,
                               info=data_node.info)
        return new_node

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        n_components = UniformIntegerHyperparameter("n_components", 2, 12, default_value=4)
        reg = CategoricalHyperparameter("reg", ["none", "shrinkage", "diagonal_fixed"], default_value="none")
        cs.add_hyperparameters([n_components, reg])
        return cs
