from mindware.components.feature_engineering.transformations.base_transformer import Transformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class EEGTransformer(Transformer):
    def __init__(self, name, random_state=1):
        super().__init__(name, random_state)

    def operate(self, data_node: EEGDataNode):
        """
        Template method that handles common EEG metadata management.
        """
        # 1. Execute the core algorithm implemented by subclasses
        new_node = self._operate(data_node)
        
        # 2. Automatically handle trans_hist (lineage tracking)
        # We copy the previous history and append the current transformer's type ID
        new_node.trans_hist = data_node.trans_hist.copy()
        new_node.trans_hist.append(self.type)

        return new_node

    def _operate(self, data_node: EEGDataNode) -> EEGDataNode:
        """
        Subclasses should implement the actual data transformation here.
        """
        raise NotImplementedError()
