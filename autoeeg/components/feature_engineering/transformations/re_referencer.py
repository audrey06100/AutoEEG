import mne
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from autoeeg.components.feature_engineering.transformations.base_eeg_transformer import EEGTransformer
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode

class ReReferencer(EEGTransformer):
    def __init__(self, method='average', robust=False, random_state=1):
        super().__init__("re-referencer", random_state)
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
                    # A simple way to do robust CAR: 
                    # detect channels with extremely high variance and mark them as bad 
                    # before calculating the average.
                    # This is a placeholder for more complex logic.
                    pass
                
                # set_eeg_reference with 'average' performs CAR in MNE
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
        # Only search for robust option if we chose ReReferencer
        # (The option of NOT re-referencing is handled by EmptyTransformer in main.py)
        robust = CategoricalHyperparameter("robust", [True, False], default_value=False)
        cs.add_hyperparameter(robust)
        return cs
