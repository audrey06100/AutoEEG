import mne
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from autoeeg.components.feature_engineering.transformations.base_eeg_transformer import EEGTransformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class Segmenter(EEGTransformer):
    def __init__(self, tmin=-0.2, tmax=0.5, baseline=None, random_state=1):
        super().__init__("segmenter", random_state)
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline # Default to None to avoid errors with positive tmin
        self.type = 104

    def _operate(self, data_node: EEGDataNode):
        raw_list = data_node.data[0] # One Raw per Subject
        event_list = data_node.data[1] # One Event array per Subject
        subject_ids = data_node.subject_ids
        
        all_epochs_list = []
        all_labels_list = []
        new_subject_ids_list = []
        
        for i, (raw, events) in enumerate(zip(raw_list, event_list)):
            # Create Epochs
            # We use the baseline passed during init (defaults to None)
            epochs = mne.Epochs(raw, events, tmin=self.tmin, tmax=self.tmax, 
                                preload=True, baseline=self.baseline, verbose=False)
            
            e_data = epochs.get_data()
            e_labels = epochs.events[:, 2]
            
            if e_data.shape[0] > 0:
                all_epochs_list.append(e_data)
                all_labels_list.append(e_labels)
                new_subject_ids_list.append(np.repeat(subject_ids[i], len(e_labels)))
        
        if not all_epochs_list:
            raise ValueError("No epochs were created! Check tmin/tmax and event markers.")

        # Time Alignment
        min_times = min(arr.shape[2] for arr in all_epochs_list)
        aligned_epochs = [arr[:, :, :min_times] for arr in all_epochs_list]
        
        # Merge all into one 3D array
        X = np.concatenate(aligned_epochs, axis=0)
        y = np.concatenate(all_labels_list, axis=0)
        final_subject_ids = np.concatenate(new_subject_ids_list, axis=0)
        
        return EEGDataNode(data=[X, y], 
                           feature_type=['numeric'] * X.shape[1], 
                           task_type=data_node.task_type, 
                           subject_ids=final_subject_ids,
                           info=raw_list[0].info)

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        tmin = UniformFloatHyperparameter("tmin", -0.5, 0.0, default_value=-0.2)
        tmax = UniformFloatHyperparameter("tmax", 0.2, 1.0, default_value=0.5)
        cs.add_hyperparameters([tmin, tmax])
        return cs
