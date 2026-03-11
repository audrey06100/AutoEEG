from autoeeg.components.feature_engineering.transformations.base_eeg_transformer import EEGTransformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class Identity(EEGTransformer):
    """
    An EEG-compatible version of EmptyTransformer.
    It simply returns the input data node without any changes,
    preserving all EEG metadata (subject_ids, info, etc.).
    """
    def __init__(self, random_state=1):
        super().__init__("identity", random_state)
        self.type = 0 # Match mindware's empty transformer type

    def _operate(self, data_node: EEGDataNode):
        # Identity operation: return a copy of the same node
        return data_node.copy_()

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        from ConfigSpace.configuration_space import ConfigurationSpace
        return ConfigurationSpace()
