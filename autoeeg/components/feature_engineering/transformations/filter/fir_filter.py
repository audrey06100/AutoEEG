import mne
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from autoeeg.components.feature_engineering.transformations.base_eeg_transformer import EEGTransformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class FIRFilter(EEGTransformer):
    def __init__(self, l_freq=1.0, h_freq=40.0, random_state=1):
        super().__init__("fir_filter", random_state)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = 'fir'
        self.type = 102

    def _operate(self, data_node: EEGDataNode):
        raw_list = data_node.data[0]
        new_raw_list = []
        for raw in raw_list:
            sfreq = raw.info['sfreq']
            # Safety checks for Nyquist
            actual_h_freq = min(self.h_freq, sfreq / 2 - 0.5)
            actual_l_freq = min(self.l_freq, actual_h_freq - 0.5)
            
            filtered_raw = raw.copy().filter(l_freq=actual_l_freq, h_freq=actual_h_freq, 
                                            method=self.method, fir_design='firwin', verbose=False)
            new_raw_list.append(filtered_raw)
            
        return EEGDataNode(data=[new_raw_list, data_node.data[1]], 
                           feature_type=data_node.feature_types, 
                           task_type=data_node.task_type, 
                           subject_ids=data_node.subject_ids,
                           info=data_node.info)

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        l_freq = UniformFloatHyperparameter("l_freq", 1.0, 15.0, default_value=7.0)
        h_freq = UniformFloatHyperparameter("h_freq", 25.0, 45.0, default_value=30.0)
        cs.add_hyperparameters([l_freq, h_freq])
        return cs
