import mne
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from autoeeg.components.feature_engineering.transformations.base_eeg_transformer import EEGTransformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class BandpassFilter(EEGTransformer):
    def __init__(self, l_freq=1.0, h_freq=40.0, method='fir', random_state=1):
        super().__init__("filter", random_state)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        self.type = 102

    def _operate(self, data_node: EEGDataNode):
        raw_list = data_node.data[0]
        
        new_raw_list = []
        for raw in raw_list:
            sfreq = raw.info['sfreq']
            
            # 1. Nyquist Check & Correction
            # h_freq must be less than sfreq / 2
            actual_h_freq = self.h_freq
            if actual_h_freq >= sfreq / 2:
                actual_h_freq = sfreq / 2 - 0.5
                
            # 2. Consistency Check
            actual_l_freq = self.l_freq
            if actual_l_freq >= actual_h_freq:
                actual_l_freq = actual_h_freq * 0.5 # Simple heuristic
            
            # Apply Filter
            # verbose=False to keep logs clean in AutoML
            # n_jobs=1 because mindware might be parallelizing already
            filtered_raw = raw.copy().filter(l_freq=actual_l_freq, h_freq=actual_h_freq, 
                                            method=self.method, verbose=False)
            new_raw_list.append(filtered_raw)
            
        new_node = EEGDataNode(data=[new_raw_list, data_node.data[1]], 
                               feature_type=data_node.feature_types, 
                               task_type=data_node.task_type, 
                               subject_ids=data_node.subject_ids,
                               info=data_node.info)
        return new_node

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        
        # Focused search space for Motor Imagery
        l_freq = UniformFloatHyperparameter("l_freq", 0.1, 15.0, default_value=1.0)
        h_freq = UniformFloatHyperparameter("h_freq", 30.0, 45.0, default_value=40.0)
        method = CategoricalHyperparameter("method", ["fir", "iir"], default_value="fir")
        
        cs.add_hyperparameters([l_freq, h_freq, method])
        return cs
