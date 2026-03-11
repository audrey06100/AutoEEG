import numpy as np
from mindware.components.feature_engineering.transformation_graph import DataNode

class EEGDataNode(DataNode):
    def __init__(self, data=None, feature_type=None, task_type=None, feature_names=None, 
                 subject_ids=None, info=None):
        """
        data[0]: List[mne.io.Raw] (One element per Subject) or 3D np.ndarray (Total Epochs)
        data[1]: List[np.ndarray] (One event array per Subject) or 1D np.ndarray (Labels)
        """
        super().__init__(data, feature_type, task_type, feature_names)
        self.subject_ids = subject_ids # If List, length == len(data[0])
        self.info = info

    @property
    def shape(self):
        if isinstance(self.data[0], list):
            # (n_subjects, n_channels)
            n_subs = len(self.data[0])
            n_channels = len(self.data[0][0].ch_names) if n_subs > 0 else 0
            return (n_subs, n_channels)
        return self.data[0].shape

    def copy_(self):
        if isinstance(self.data[0], list):
            new_data_0 = list(self.data[0])
            new_data_1 = list(self.data[1]) if self.data[1] is not None else None
        else:
            new_data_0 = self.data[0].copy()
            new_data_1 = self.data[1].copy() if self.data[1] is not None else None
        
        new_data = [new_data_0, new_data_1]
        new_node = EEGDataNode(new_data, 
                               self.feature_types.copy() if self.feature_types is not None else None, 
                               self.task_type,
                               self.feature_names.copy() if self.feature_names is not None else None,
                               subject_ids=list(self.subject_ids) if self.subject_ids is not None else None,
                               info=self.info)
        
        new_node.trans_hist = self.trans_hist.copy()
        new_node.depth = self.depth
        new_node.enable_balance = self.enable_balance
        new_node.data_balance = self.data_balance
        new_node.config = self.config
        return new_node
