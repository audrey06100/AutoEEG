import mne
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from autoeeg.components.feature_engineering.transformations.base_eeg_transformer import EEGTransformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class CARTransformer(EEGTransformer):
    def __init__(self, method='average', robust=False, random_state=1):
        super().__init__("car_transformer", random_state)
        self.method = method
        self.robust = robust
        self.type = 103

    def _operate(self, data_node: EEGDataNode):
        raw_list = data_node.data[0]
        
        new_raw_list = []
        for raw in raw_list:
            # We copy to avoid side effects
            raw_cp = raw.copy()
            
            if self.method == 'average':
                if self.robust:
                    # Robust CAR: Detect channels with extreme variance and mark as bad
                    # before calculating the average.
                    data = raw_cp.get_data(picks='eeg')
                    # Calculate standard deviation for each channel
                    stds = data.std(axis=1)
                    median_std = np.median(stds)
                    std_std = np.std(stds)
                    
                    # Heuristic: channels with std > median + 2.5 * std_std are likely bad
                    # (Standard outlier detection)
                    threshold = median_std + 2.5 * std_std
                    bad_idx = np.where(stds > threshold)[0]
                    
                    eeg_ch_names = [raw_cp.ch_names[i] for i in raw_cp.ch_info['picks']] if hasattr(raw_cp, 'ch_info') else raw_cp.copy().pick_types(eeg=True).ch_names
                    
                    new_bads = [eeg_ch_names[i] for i in bad_idx]
                    raw_cp.info['bads'] = list(set(raw_cp.info['bads'] + new_bads))
                
                # set_eeg_reference with 'average' performs CAR in MNE
                # It automatically excludes 'bads' from the average calculation
                raw_cp.set_eeg_reference(ref_channels='average', verbose=False)
            
            new_raw_list.append(raw_cp)
            
        new_node = EEGDataNode(data=[new_raw_list, data_node.data[1]], 
                               feature_type=data_node.feature_types, 
                               task_type=data_node.task_type, 
                               subject_ids=data_node.subject_ids,
                               info=data_node.info)
        return new_node

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        # Only search for robust option if we chose CARTransformer
        # (The option of NOT re-referencing is handled by EmptyTransformer in main.py)
        robust = CategoricalHyperparameter("robust", [True, False], default_value=False)
        cs.add_hyperparameter(robust)
        return cs
