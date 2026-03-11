import mne
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from autoeeg.components.feature_engineering.transformations.base_eeg_transformer import EEGTransformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class Resampler(EEGTransformer):
    def __init__(self, sfreq=128, random_state=1):
        super().__init__("resampler", random_state)
        self.sfreq = sfreq
        self.type = 101

    def _operate(self, data_node: EEGDataNode):
        raw_list = data_node.data[0]
        event_list = data_node.data[1]
        
        new_raw_list = []
        new_event_list = []
        
        for raw, events in zip(raw_list, event_list):
            # mne.io.Raw.resample can return updated events if provided
            # We copy to avoid modifying the original Raw in-place (important for AutoML)
            resampled_raw = raw.copy().resample(sfreq=self.sfreq, events=events)
            
            # If events were provided, resample returns (raw, events)
            if isinstance(resampled_raw, tuple):
                new_raw_list.append(resampled_raw[0])
                new_event_list.append(resampled_raw[1])
            else:
                new_raw_list.append(resampled_raw)
                new_event_list.append(events) # Should not happen if events passed
        
        new_node = EEGDataNode(data=[new_raw_list, new_event_list], 
                               feature_type=data_node.feature_types, 
                               task_type=data_node.task_type, 
                               subject_ids=data_node.subject_ids,
                               info=data_node.info)
        return new_node

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        sfreq = UniformIntegerHyperparameter("sfreq", 256, 512, default_value=128)
        cs.add_hyperparameter(sfreq)
        return cs
