from mindware.components.models.base_searcher import BaseSearcher
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from mindware.components.models.search.nas_utils.nb1_utils import bench101_opt_choices

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

NUM_WORKERS = 10


class NASBench101Searcher(BaseSearcher):
    def __init__(self, arch_config, batch_size=128,
                 epoch_num=108, learning_rate=3e-4,
                 weight_decay=1e-4, lr_decay=1e-1,
                 random_state=None, grayscale=False,
                 device='cpu', **kwargs):
        super().__init__(arch_config=arch_config,
                         batch_size=batch_size,
                         epoch_num=epoch_num,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         lr_decay=lr_decay,
                         random_state=random_state,
                         grayscale=grayscale,
                         device=device,
                         **kwargs)

        self.space = 'nasbench101'

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'NB101Searcher',
                'name': 'NASBench101Searcher',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        vertex = 7
        cs = ConfigurationSpace()
        total_edges = sum(list(range(vertex)))
        for e in range(0, total_edges):
            cs.add_hyperparameter(CategoricalHyperparameter('edge_%d' % e, choices=[0, 1]))

        for v in range(1, vertex - 1):
            cs.add_hyperparameter(CategoricalHyperparameter('vertex_%d' % v, choices=bench101_opt_choices))

        return cs
