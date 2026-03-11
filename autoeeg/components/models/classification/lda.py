from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mindware.components.models.base_model import BaseClassificationModel
from mindware.components.utils.constants import DENSE, UNSIGNED_DATA, PREDICTIONS
from mindware.components.utils.model_util import softmax
from mindware.components.utils.configspace_utils import check_none

class LDA(BaseClassificationModel):
    def __init__(self, shrinkage='None', shrinkage_factor=0.5, n_components=None, tol=1e-4, random_state=None):
        self.shrinkage = shrinkage
        self.shrinkage_factor = shrinkage_factor
        self.n_components = n_components
        self.tol = tol
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        if check_none(self.shrinkage):
            s_val = None
            solver = 'svd'
        elif self.shrinkage == 'auto':
            s_val = 'auto'
            solver = 'lsqr'
        else: # manual
            s_val = float(self.shrinkage_factor)
            solver = 'lsqr'

        self.estimator = LinearDiscriminantAnalysis(
            shrinkage=s_val,
            solver=solver,
            n_components=self.n_components,
            tol=self.tol
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return softmax(self.estimator.predict_proba(X))

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LDA',
                'name': 'EEG Specialized LDA',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = ConfigurationSpace()
        shrinkage = CategoricalHyperparameter("shrinkage", ["None", "auto", "manual"], default_value="auto")
        shrinkage_factor = UniformFloatHyperparameter("shrinkage_factor", 0., 1., 0.5)
        
        cs.add_hyperparameters([shrinkage, shrinkage_factor])
        cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))
        return cs
